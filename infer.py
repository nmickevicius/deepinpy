from test_tube import HyperOptArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger

import matplotlib.pyplot as plt
import os
import pathlib
import argparse

import sigpy

import h5py
import deepinpy.utils.complex as cp

from deepinpy.recons import CGSenseRecon, MoDLRecon, ResNetRecon, DeepBasisPursuitRecon

import torch
torch.backends.cudnn.enabled = True

import numpy.random
import numpy as np

def main_infer(args):

    if args.recon == 'cgsense':
        MyRecon = CGSenseRecon
    elif args.recon == 'modl':
        MyRecon = MoDLRecon
    elif args.recon == 'resnet':
        MyRecon = ResNetRecon
    elif args.recon == 'dbp':
        MyRecon = DeepBasisPursuitRecon

    # load from checkpoint
    print('loading checkpoint: {}'.format(args.checkpoint_init))
    M = MyRecon.load_from_checkpoint(args.checkpoint_init)

    # create a dataset
    # M.D = MultiChannelMRIDataset()

    # get number of datasets
    useBatchsizeOne = True
    if useBatchsizeOne:
        with h5py.File(args.data_file, 'r') as F:
            imgs = np.array(F['imgs'], dtype=np.complex)
        ndatasets = imgs.shape[0]
        for idx in range(ndatasets):

            print('%i/%i'%(idx,ndatasets))

            # load data
            with h5py.File(args.data_file, 'r') as F:
                imgs = np.array(F['imgs'][idx,...], dtype=np.complex)
                maps = np.array(F['maps'][idx,...], dtype=np.complex)
                ksp = np.array(F['ksp'][idx,...], dtype=np.complex)
                masks = np.array(F['masks'][idx,...], dtype=np.float)
                if args.multiband:
                    phi = np.array(F['phi'][idx,...],dtype=np.complex)
            if len(imgs.shape) == 3 and args.multiband:
                imgs, maps, masks, ksp = imgs[None,...], maps[None,...], masks[None,...], ksp[None,...]
                if args.multiband:
                    phi = phi[None,...]
            elif len(imgs.shape) == 2:
                imgs, maps, masks, ksp = imgs[None,...], maps[None,...], masks[None,...], ksp[None,...]

            imgs_torch_pre = cp.c2r(imgs).astype(np.float32)
            maps_torch_pre = cp.c2r(maps).astype(np.float32)
            masks_torch_pre = masks.astype(np.float32)
            ksp_torch_pre = cp.c2r(ksp).astype(np.float32)
            if args.multiband:
                phi_torch_pre = cp.c2r(phi).astype(np.float32)

            print(imgs.shape)
            print(maps.shape)
            print(masks.shape)
            print(ksp.shape)
            if args.multiband:
                print(phi.shape)
            print('')
            print(imgs_torch_pre.shape)
            print(maps_torch_pre.shape)
            print(masks_torch_pre.shape)
            print(ksp_torch_pre.shape)
            if args.multiband:
                print(phi_torch_pre.shape)


            # store a data dictionary in memory
            if args.multiband:
                data = {
                    'imgs': sigpy.to_pytorch(imgs_torch_pre),
                    'maps': sigpy.to_pytorch(maps_torch_pre),
                    'masks': sigpy.to_pytorch(masks_torch_pre),
                    'phi': sigpy.to_pytorch(phi_torch_pre),
                    'out': sigpy.to_pytorch(ksp_torch_pre)
                    }
            else:
                data = {
                    'imgs': sigpy.to_pytorch(imgs_torch_pre),
                    'maps': sigpy.to_pytorch(maps_torch_pre),
                    'masks': sigpy.to_pytorch(masks_torch_pre),
                    'out': sigpy.to_pytorch(ksp_torch_pre)
                    }

            # batch this data
            M.batch(data)

            # predict output
            output = sigpy.from_pytorch(M(sigpy.to_pytorch(ksp_torch_pre)))

            # complexify
            output = output[...,0] + 1j*output[...,1]
            output = output[0,...]
            # if args.multiband:
            #     output = output[0,::,::,::,0] + 1j*output[0,::,::,::,1]
            # else:
            #     output = output[0,::,::,0] + 1j*output[0,::,::,1]

            # allocate output
            if idx == 0:
                if args.multiband:
                    pred = np.zeros((ndatasets,output.shape[0],output.shape[1],output.shape[2]),dtype=output.dtype)
                else:
                    pred = np.zeros((ndatasets,output.shape[0],output.shape[1]),dtype=output.dtype)

            if args.multiband:
                plt.figure()
                for slc in range(output.shape[0]):
                    plt.subplot(1,output.shape[0],slc+1)
                    plt.imshow(np.abs(output[slc,::,::]),cmap='gray')
                plt.subplots_adjust(wspace=0)
                plt.show()
            else:
                plt.figure()
                plt.imshow(np.abs(output),cmap='gray')
                plt.show()

            # store in output
            if args.multiband:
                pred[idx,::,::,::] = output
            else:
                pred[idx,::,::] = output

        np.savez('inference.npz',pred=pred)

    else:

        print('Reading from input file')
        with h5py.File(args.data_file, 'r') as F:
            imgs = np.array(F['imgs'], dtype=np.complex)
            maps = np.array(F['maps'], dtype=np.complex)
            ksp = np.array(F['ksp'], dtype=np.complex)
            masks = np.array(F['masks'], dtype=np.float)
            if args.multiband:
                phi = np.array(F['phi'],dtype=np.complex)
        if len(masks.shape) == 2:
            imgs, maps, masks, ksp = imgs[None,...], maps[None,...], masks[None,...], ksp[None,...]
            if args.multiband:
                phi = phi[None,...]

        imgs_torch_pre = cp.c2r(imgs).astype(np.float32)
        maps_torch_pre = cp.c2r(maps).astype(np.float32)
        masks_torch_pre = masks.astype(np.float32)
        ksp_torch_pre = cp.c2r(ksp).astype(np.float32)
        if args.multiband:
            phi_torch_pre = cp.c2r(phi).astype(np.float32)

        print(imgs.shape)
        print(maps.shape)
        print(masks.shape)
        print(ksp.shape)
        if args.multiband:
            print(phi.shape)

        print(imgs_torch_pre.shape)
        print(maps_torch_pre.shape)
        print(masks_torch_pre.shape)
        print(ksp_torch_pre.shape)
        if args.multiband:
            print(phi_torch_pre.shape)

        print('Writing to dictionary')
        if args.multiband:
            data = {
                    'imgs': sigpy.to_pytorch(imgs_torch_pre),
                    'maps': sigpy.to_pytorch(maps_torch_pre),
                    'masks': sigpy.to_pytorch(masks_torch_pre),
                    'phi': sigpy.to_pytorch(phi_torch_pre),
                    'out': sigpy.to_pytorch(ksp_torch_pre)
            }
        else:
            data = {
                    'imgs': sigpy.to_pytorch(imgs_torch_pre),
                    'maps': sigpy.to_pytorch(maps_torch_pre),
                    'masks': sigpy.to_pytorch(masks_torch_pre),
                    'out': sigpy.to_pytorch(ksp_torch_pre)
            }
        print('Preparing data batch')
        M.batch(data)

        print('Calling M(y)')
        pred = M(y)

        print(pred.shape)
        np.savez('inference.npz',pred=pred)

    return pred


if __name__ == '__main__':
    usage_str = 'usage: %(prog)s [options]'
    description_str = 'deep inverse problems optimization'

    parser = HyperOptArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter, strategy='random_search')

    parser.opt_range('--step', type=float, dest='step', default=.001, help='step size/learning rate', tunable=True, nb_samples=100, low=.0001, high=.001)
    parser.opt_range('--l2lam_init', action='store', type=float, dest='l2lam_init', default=.001, tunable=False, low=.0001, high=100, help='initial l2 regularization')
    parser.opt_list('--solver', action='store', dest='solver', type=str, tunable=False, options=['sgd', 'adam'], help='optimizer/solver ("adam", "sgd")', default="sgd")
    parser.opt_range('--cg_max_iter', action='store', dest='cg_max_iter', type=int, tunable=False, low=1, high=20, help='max number of conjgrad iterations', default=10)
    parser.opt_range('--batch_size', action='store', dest='batch_size', type=int, tunable=False, low=1, high=20, help='batch size', default=2)
    parser.opt_range('--num_unrolls', action='store', dest='num_unrolls', type=int, tunable=False, low=1, high=10, nb_samples=4, help='number of unrolls', default=4)
    parser.opt_range('--num_admm', action='store', dest='num_admm', type=int, tunable=False, low=1, high=10, nb_samples=4, help='number of ADMM iterations', default=3)
    parser.opt_list('--network', action='store', dest='network', type=str, tunable=False, options=['ResNet', 'ResNet5Block'], help='which denoiser network to use', default='ResNet')
    parser.opt_list('--latent_channels', action='store', dest='latent_channels', type=int, tunable=False, options=[16, 32, 64, 128], help='number of latent channels', default=64)
    parser.opt_range('--num_blocks', action='store', dest='num_blocks', type=int, tunable=False, low=1, high=4, nb_samples=3, help='number of ResNetBlocks', default=3)
    parser.opt_range('--dropout', action='store', dest='dropout', type=float, tunable=False, low=0., high=.5, help='dropout fraction', default=0.)
    parser.opt_list('--batch_norm', action='store_true', dest='batch_norm', tunable=False, options=[True, False], help='batch normalization', default=False)

    parser.add_argument('--num_accumulate', action='store', dest='num_accumulate', type=int, help='nunumber of batch accumulations', default=1)
    parser.add_argument('--name', action='store', dest='name', type=str, help='experiment name', default=1)
    parser.add_argument('--version', action='store', dest='version', type=int, help='version number', default=None)
    parser.add_argument('--gpu', action='store', dest='gpu', type=str, help='gpu number(s)', default=None)
    parser.add_argument('--cpu', action='store_true', dest='cpu', help='Use CPU', default=False)
    parser.add_argument('--num_epochs', action='store', dest='num_epochs', type=int, help='number of epochs', default=20)
    parser.add_argument('--random_seed', action='store', dest='random_seed', type=int, help='random number seed for numpy', default=723)
    parser.add_argument('--recon', action='store', type=str, dest='recon', default='cgsense', help='reconstruction method')
    parser.add_argument('--data_file', action='store', dest='data_file', type=str, help='data.h5', default=None)
    parser.add_argument('--data_val_file', action='store', dest='data_val_file', type=str, help='data.h5', default=None)
    parser.add_argument('--num_data_sets', action='store', dest='num_data_sets', type=int, help='number of data sets to use', default=None)
    parser.add_argument('--multiband', action='store', dest='multiband', help='Multiband factor', default=None) # NJM: support for multiband
    parser.add_argument('--num_workers', action='store', type=int,  dest='num_workers', help='number of workers', default=0)
    parser.add_argument('--shuffle', action='store_true', dest='shuffle', help='shuffle input data files each epoch', default=False)
    parser.add_argument('--clip_grads', action='store', type=float, dest='clip_grads', help='clip gradients to [-val, +val]', default=False)
    parser.add_argument('--cg_eps', action='store', type=float, dest='cg_eps', help='conjgrad eps', default=1e-4)
    parser.add_argument('--stdev', action='store', type=float, dest='stdev', help='complex valued noise standard deviation', default=.01)
    parser.add_argument('--max_norm_constraint', action='store', type=float, dest='max_norm_constraint', help='norm constraint on weights', default=None)
    parser.add_argument('--fully_sampled', action='store_true', dest='fully_sampled', help='fully_sampled', default=False)
    parser.add_argument('--adam_eps', action='store', type=float, dest='adam_eps', help='adam epsilon', default=1e-8)
    parser.add_argument('--inverse_crime', action='store_true', dest='inverse_crime', help='inverse crime', default=False)
    parser.add_argument('--use_sigpy', action='store_true', dest='use_sigpy', help='use SigPy for Linops', default=False)
    parser.add_argument('--noncart', action='store_true', dest='noncart', help='NonCartesian data', default=False)
    parser.add_argument('--abs_loss', action='store_true', dest='abs_loss', help='use magnitude for loss', default=False)
    parser.add_argument('--self_supervised', action='store_true', dest='self_supervised', help='self-supervised loss', default=False)
    parser.add_argument('--hyperopt', action='store_true', dest='hyperopt', help='perform hyperparam optimization', default=False)
    parser.add_argument('--checkpoint_init', action='store', dest='checkpoint_init', type=str, help='load from checkpoint', default=None)
    parser.json_config('--config', default=None)

    args = parser.parse_args()

    pred = main_infer(args)
