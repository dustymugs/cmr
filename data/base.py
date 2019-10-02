"""
Base data loading class.

Should output:
    - img: B X 3 X H X W
    - kp: B X nKp X 2
    - mask: B X H X W
    - sfm_pose: B X 7 (s, tr, q)
    (kp, sfm_pose) correspond to image coordinates in [-1, 1]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np

import scipy.io as sio
import scipy.linalg
import scipy.ndimage.interpolation
import skimage
from absl import flags, app

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from ..utils import image as image_utils
from ..utils import transformations

__all__ = [
   'BaseDataset',
   'base_loader'
]

# -------------- flags ------------- #
# ---------------------------------- #
    
# directory containing the "images" directory
flags.DEFINE_string('data_dir', '', 'Data Directory')
# directory containing the "data" and "sfm" directories
flags.DEFINE_string('cache_dir', '', 'Cache Directory')

flags.DEFINE_integer('img_size', 256, 'image size')

flags.DEFINE_float('padding_frac', 0.05,
                   'bbox is increased by this fraction of max_dim')

flags.DEFINE_float('jitter_frac', 0.05,
                   'bbox is jittered by this fraction of max_dim')

flags.DEFINE_enum('split', 'train', ['train', 'val', 'all', 'test'], 'eval split')
#flags.DEFINE_integer('num_kps', 15, 'The dataloader should override these.')
flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')


# -------------- Dataset ------------- #
# ------------------------------------ #
class _BaseDataset(Dataset):
    ''' 
    img, mask, kp, pose data loader

    DO NOT INHERIT FROM THIS!!! USE BaseDataset BELOW!!!
    '''

    def __init__(self, opts, filter_key=None):
        # Child class should define/load:
        # self.kp_perm
        # self.img_dir
        # self.anno
        # self.anno_sfm
        self.opts = opts
        self.img_size = opts.img_size
        self.jitter_frac = opts.jitter_frac
        self.padding_frac = opts.padding_frac
        self.filter_key = filter_key
    
    def forward_img(self, index):
        data = self.anno[index]
        data_sfm = self.anno_sfm[index]

        # sfm_pose = (sfm_c, sfm_t, sfm_r)
        sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]

        sfm_rot = np.pad(sfm_pose[2], (0,1), 'constant')
        sfm_rot[3, 3] = 1
        sfm_pose[2] = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)

        img_path = osp.join(self.img_dir, str(data.rel_path))
        img = skimage.io.imread(img_path) / 255.0
        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        mask = np.expand_dims(data.mask, 2)

        # Adjust to 0 indexing
        bbox = np.array(
            [data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2],
            float) - 1

        parts = data.parts.T.astype(float)
        kp = np.copy(parts)
        vis = kp[:, 2] > 0
        kp[vis, :2] -= 1

        # Peturb bbox
        if self.opts.split == 'train':
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=self.jitter_frac)
        else:
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=0)
        bbox = image_utils.square_bbox(bbox)

        # crop image around bbox, translate kps
        img, mask, kp, sfm_pose = self.crop_image(img, mask, bbox, kp, vis, sfm_pose)

        # scale image, and mask. And scale kps.        
        img, mask, kp, sfm_pose = self.scale_image(img, mask, kp, vis, sfm_pose)

        # Mirror image on random.
        if self.opts.split == 'train':
            img, mask, kp, sfm_pose = self.mirror_image(img, mask, kp, sfm_pose)

        # Normalize kp to be [-1, 1]
        img_h, img_w = img.shape[:2]
        kp_norm, sfm_pose = self.normalize_kp(kp, sfm_pose, img_h, img_w)

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))

        return data.rel_path, img, kp_norm, mask, sfm_pose

    def normalize_kp(self, kp, sfm_pose, img_h, img_w):
        vis = kp[:, 2, None] > 0
        new_kp = np.stack([2 * (kp[:, 0] / img_w) - 1,
                           2 * (kp[:, 1] / img_h) - 1,
                           kp[:, 2]]).T
        sfm_pose[0] *= (1.0/img_w + 1.0/img_h)
        sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1
        new_kp = vis * new_kp

        return new_kp, sfm_pose

    def crop_image(self, img, mask, bbox, kp, vis, sfm_pose):
        # crop image and mask and translate kps
        img = image_utils.crop(img, bbox, bgval=1)
        mask = image_utils.crop(mask, bbox, bgval=0)
        kp[vis, 0] -= bbox[0]
        kp[vis, 1] -= bbox[1]
        sfm_pose[1][0] -= bbox[0]
        sfm_pose[1][1] -= bbox[1]
        return img, mask, kp, sfm_pose

    def scale_image(self, img, mask, kp, vis, sfm_pose):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.img_size / float(max(bwidth, bheight))
        img_scale, _ = image_utils.resize_img(img, scale)
        # if img_scale.shape[0] != self.img_size:
        #     print('bad!')
        #     import ipdb; ipdb.set_trace()
        mask_scale, _ = image_utils.resize_img(mask, scale)
        kp[vis, :2] *= scale
        sfm_pose[0] *= scale
        sfm_pose[1] *= scale

        return img_scale, mask_scale, kp, sfm_pose

    def mirror_image(self, img, mask, kp, sfm_pose):
        kp_perm = self.kp_perm
        if np.random.rand(1) > 0.5:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()
            
            # Flip kps.
            new_x = img.shape[1] - kp[:, 0] - 1
            kp_flip = np.hstack((new_x[:, None], kp[:, 1:]))
            kp_flip = kp_flip[kp_perm, :]
            # Flip sfm_pose Rot.
            R = transformations.quaternion_matrix(sfm_pose[2])
            flip_R = np.diag([-1, 1, 1, 1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
            sfm_pose[2] = transformations.quaternion_from_matrix(flip_R, isprecise=True)
            # Flip tx
            tx = img.shape[1] - sfm_pose[1][0] - 1
            sfm_pose[1][0] = tx
            return img_flip, mask_flip, kp_flip, sfm_pose
        else:
            return img, mask, kp, sfm_pose

    @property
    def kp_perm(self):
        return getattr(self, '_kp_perm', np.array([]))

    @kp_perm.setter
    def kp_perm(self, new_values):
        self._kp_perm = new_values

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        rel_path, img, kp, mask, sfm_pose = self.forward_img(index)
        sfm_pose[0].shape = 1

        elem = {
            'rel_path': rel_path,
            'img': img,
            'kp': kp,
            'mask': mask,
            'sfm_pose': np.concatenate(sfm_pose),
            'inds': index,
        }

        if self.filter_key is not None:
            assert self.filter_key in elem.keys(), \
                'Bad filter key {}'.format(self.filter_key)
            if self.filter_key == 'sfm_pose':
                # Return both vis and sfm_pose
                vis = elem['kp'][:, 2]
                elem = {
                    'vis': vis,
                    'sfm_pose': elem['sfm_pose'],
                }
            else:
                elem = elem[self.filter_key]

        return elem

class BaseDataset(_BaseDataset):
    '''
    Generic Data loader
    '''

    def __init__(self, opts, filter_key=None):
        super().__init__(opts, filter_key=filter_key)
        self.filter_key = filter_key

        self.data_dir = opts.data_dir
        self.cache_dir = opts.cache_dir

        self.img_dir = osp.join(self.data_dir, 'images')
        self.anno_path = osp.join(
            self.cache_dir,
            'data',
            '{}_cleaned.mat'.format(opts.split)
        )
        self.anno_sfm_path = osp.join(
            self.cache_dir,
            'sfm',
            'anno_{}.mat'.format(opts.split)
        )

        assert osp.exists(self.anno_path), \
            '{} not found'.format(self.anno_path)

        # Load the annotation file.
        print('Loading {}'.format(self.anno_path))
        self.anno = sio.loadmat(
            self.anno_path,
            struct_as_record=False,
            squeeze_me=True
        )['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path,
            struct_as_record=False,
            squeeze_me=True
        )['sfm_anno']

        self.num_imgs = len(self.anno)
        print('{} images'.format(self.num_imgs))

        # assume that kp_perm is sequentially arranged (e.g. 0, 1, 2, 3 vs 0, 2, 1, 3)
        # override this in inherited class if needed
        self.kp_perm = np.array(list(range(self.anno[0].parts.shape[-1])))

# ------------ Data Loader ----------- #
# ------------------------------------ #
def base_loader(d_set_func, batch_size, opts, filter_key=None, shuffle=True):
    dset = d_set_func(opts, filter_key=filter_key)
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        drop_last=True
    )
