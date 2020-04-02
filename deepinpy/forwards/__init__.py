"""
Deep inverse problems in Python

forwards submodule
A Forward object takes an input image x and returns measurements y
"""

from .mcmri.mcmri import MultiChannelMRI
from .mcmri.dataset import MultiChannelMRIDataset
from .mbmri.mbmri import MultiBandMRI # NJM
from .mbmri.dataset import MultiBandMRIDataset # NJM
