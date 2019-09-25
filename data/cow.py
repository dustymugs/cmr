from __future__ import absolute_import

import numpy as np

from .base import base_loader, BaseDataset

__all__ = [
    'CowDataset'
]

# -------------- Dataset ------------- #
# ------------------------------------ #
class CowDataset(BaseDataset):
    '''
    Cow Data loader
    '''

    def __init__(self, opts, filter_key=None):
        super().__init__(opts=opts, filter_key=filter_key)
        self.img_dir = '' # note that we override the default
        self.kp_perm = np.array(
            [1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21]
        ) - 1

#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts, shuffle=True):
    return base_loader(CowDataset, opts.batch_size, opts, filter_key=None, shuffle=shuffle)


def kp_data_loader(batch_size, opts):
    return base_loader(CowDataset, batch_size, opts, filter_key='kp')


def mask_data_loader(batch_size, opts):
    return base_loader(CowDataset, batch_size, opts, filter_key='mask')

    
def sfm_data_loader(batch_size, opts):
    return base_loader(CowDataset, batch_size, opts, filter_key='sfm_pose')
