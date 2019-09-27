"""
CUB has 11788 images total, for 200 subcategories.
5994 train, 5794 test images.

After removing images that are truncated:
min kp threshold 6: 5964 train, 5771 test.
min_kp threshold 7: 5937 train, 5747 test.

"""
from __future__ import absolute_import

import os.path as osp
import numpy as np

import scipy.io as sio
from absl import flags, app

import torch
from torch.utils.data import Dataset

from .base import base as base_loader, BaseDataset

# -------------- Dataset ------------- #
# ------------------------------------ #
class BirdDataset(BaseDataset):
    '''
    Bird Data loader
    '''

    def __init__(self, opts, filter_key=None):
        super().__init__(opts=opts, filter_key=filter_key)
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1;

#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts, shuffle=True):
    return base_loader(BirdDataset, opts.batch_size, opts, filter_key=None, shuffle=shuffle)


def kp_data_loader(batch_size, opts):
    return base_loader(BirdDataset, batch_size, opts, filter_key='kp')


def mask_data_loader(batch_size, opts):
    return base_loader(BirdDataset, batch_size, opts, filter_key='mask')

    
def sfm_data_loader(batch_size, opts):
    return base_loader(BirdDataset, batch_size, opts, filter_key='sfm_pose')
