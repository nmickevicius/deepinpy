#!/usr/bin/env python

import numpy as np
import torch

import deepinpy.utils.complex as cp

class MultiBandMRI(torch.nn.Module):

    def __init__(self, maps, mask, phi, l2lam=False, ksp_shape=None, img_shape=None, use_sigpy=True, noncart=True):

        super(MultiBandMRI, self).__init__()
        self.maps = maps
        self.mask = mask
        self.phi = phi
        self.l2lam = l2lam
        self.ksp_shape = ksp_shape
        self.img_shape = img_shape
        self.noncart = noncart
        self._normal = None

        assert self.noncart, 'Non-cartesian implementation only for now'
        assert use_sigpy, 'Must use SigPy for NUFFT'

        if use_sigpy:

            from sigpy import from_pytorch, to_device, Device
            sp_device = Device(0) #Device(self.maps.device.index) # NJM: change to Device(-1) for CPU

            # from real-valued pytorch tensor of shape (batchsize,coils,SMS,N,N,2)
            # to complex-valued numpy/cupy array of shape (batchsize,coils,SMS,N,N)
            self.maps = to_device(from_pytorch(self.maps, iscomplex=True), device=sp_device)

            # from real-valued pytorch tensor of shape (batchsize,nspokes,nreadout,2)
            # to real-valued numpy/cupy array of same shape
            self.mask = to_device(from_pytorch(self.mask, iscomplex=False), device=sp_device)

            # from real-valued pytorch tensor of shape (batchsize,SMS,nspokes,nreadout,2)
            # to complex-valued numpy/cupy array of shape (batchsize,SMS,nspokes,nreadout)
            self.phi = to_device(from_pytorch(self.phi, iscomplex=True), device=sp_device)

            self.img_shape = self.img_shape[:-1] # convert R^2N to C^N
            self.ksp_shape = self.ksp_shape[:-1] # convert R^2N to C^N

            self._build_model_sigpy()

    def _build_model_sigpy(self):

        from sigpy.linop import Multiply, Sum
        if self.noncart:
            from sigpy.linop import NUFFT, NUFFTAdjoint
        else:
            from sigpy.linop import FFT
        from sigpy import to_pytorch_function

        if self.noncart:

            Aop_list = []
            Aop_adjoint_list = []
            Aop_normal_list = []

            _img_shape = self.img_shape[1:] # (SMS, N, N)
            _ksp_shape = self.ksp_shape[1:] # (coils, SMS, spokes, readout)

            # construct linear encoding operators for each item in batch size
            for i in range(self.img_shape[0]):

                # get coil maps, trajectory, and phase modulation information
                # for the current batch item
                _maps = self.maps[i, ...] # (coils, SMS, N, N)
                _mask = self.mask[i, ...] # (spokes, readout, 2)
                _phi = self.phi[i, ...]   # (coils, SMS, spokes, readout)

                # coil sensitivity map operator
                C = Multiply( _img_shape, _maps)

                # non-uniform fast Fourier transform operator
                F = NUFFT( _maps.shape, _mask)

                # phase modulation operator
                P = Multiply( _ksp_shape, _phi)

                # sum over slices operator
                S = Sum( _ksp_shape, axes=(1,))

                # forward encoding operator
                A = S*P*F*C

                # adjoint encoding operator
                AH = C.H * F.H * P.H * S.H

                Aop_list.append(to_pytorch_function(A, input_iscomplex=True, output_iscomplex=True).apply)
                Aop_adjoint_list.append(to_pytorch_function(AH, input_iscomplex=True, output_iscomplex=True).apply)
                Aop_normal_list.append(to_pytorch_function(AH * A, input_iscomplex=True, output_iscomplex=True).apply)

            self.Aop_list = Aop_list
            self.Aop_adjoint_list = Aop_adjoint_list
            self.Aop_normal_list = Aop_normal_list
            self._forward = self._nufft_batch_forward
            self._adjoint = self._nufft_batch_adjoint
            self._normal = self._nufft_batch_normal

    def _nufft_batch_forward(self, x):
        batch_size = x.shape[0]
        out0 = self.Aop_list[0](x[0])
        if batch_size == 1:
            return out0[None,...]
        else:
            out = out0
            for i in range(1, batch_size):
                out = torch.stack((out, self.Aop_list[i](x[i])), axis=0)
            return out

    def _nufft_batch_adjoint(self, x):
        batch_size = x.shape[0]
        out0 = self.Aop_adjoint_list[0](x[0])
        if batch_size == 1:
            return out0[None,...]
        else:
            out = out0
            for i in range(1, batch_size):
                out = torch.stack((out, self.Aop_adjoint_list[i](x[i])), axis=0)
            return out

    def _nufft_batch_normal(self, x):
        batch_size = x.shape[0]
        out0 = self.Aop_normal_list[0](x[0])
        if batch_size == 1:
            return out0[None,...]
        else:
            out = out0
            for i in range(1, batch_size):
                out = torch.stack((out, self.Aop_normal_list[i](x[i])), axis=0)
            return out

    def _forward(self, x):
        return sense_forw(x, self.maps, self.mask)

    def _adjoint(self, y):
        return sense_adj(y, self.maps, self.mask)

    def forward(self, x):
        return self._forward(x)

    def adjoint(self, y):
        return self._adjoint(y)

    def normal(self, x):
        if self._normal:
            out = self._normal(x)
        else:
            out = self.adjoint(self.forward(x))
        if self.l2lam:
            out = out + self.l2lam * x
        return out

    #def normal(self, x):
        #return self.normal_fun(x)

def maps_forw(img, maps):
    return cp.zmul(img[:,None,:,:,:], maps)

def maps_adj(cimg, maps):
    return torch.sum(cp.zmul(cp.zconj(maps), cimg), 1, keepdim=False)

def fft_forw(x, ndim=2):
    return torch.fft(x, signal_ndim=ndim, normalized=True)

def fft_adj(x, ndim=2):
    return torch.ifft(x, signal_ndim=ndim, normalized=True)

def mask_forw(y, mask):
    return y * mask[:,None,:,:,None]

def sense_forw(img, maps, mask):
    return mask_forw(fft_forw(maps_forw(img, maps)), mask)

def sense_adj(ksp, maps, mask):
    return maps_adj(fft_adj(mask_forw(ksp, mask)), maps)

def sense_normal(img, maps, mask):
    return maps_adj(fft_adj(mask_forw(fft_forw(maps_forw(img, maps)), mask)), maps)
