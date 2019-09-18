import click
from contextlib import closing
import copy
import math
import numpy as np
import os
import pandas as pd
import random
import scipy.io as sio
import tables as tb
import time

class DynamicRecArray(object):
    def __init__(self, dtype, size=10):
        self.dtype = np.dtype(dtype) if not isinstance(dtype, np.dtype) else dtype
        self.length = 0
        self.size = size
        self._data = np.empty(self.size, dtype=self.dtype)
        self._not_deleted = np.ones(self.size, dtype=bool)

    def __len__(self):
        return self.length

    def _resize(self):

        if self.length == self.size:
            self.size = int(1.5 * self.size)
            self._data = np.resize(self._data, self.size)
            self._not_deleted = np.resize(self._not_deleted, self.size)

    def append(self, rec):
        self._resize()
        self._data[self.length] = rec
        self.length += 1

    def extend(self, recs):
        for rec in recs:
            self.append(rec)

    def delete(self, index):

        self._not_deleted[index] = False

    @property
    def data(self):
        return (self._data[self._not_deleted])[:self.length]

class AnnotationManager(object):

    STRUCTURED_DTYPES = {
        'images': np.dtype([
            ('rel_path', 'O'),
            ('bbox', 'O'),
            ('parts', 'O'),
            ('mask', 'O'),
        ]),
        'bbox': np.dtype([
            ('x1', 'O'),
            ('y1', 'O'),
            ('x2', 'O'),
            ('y2', 'O'),
        ])
    }

    def _get_attribute(self, kw):
        assert hasattr(self, kw)
        return getattr(self, kw)

    def _set_attribute(self, kw, value):
        setattr(self, kw, value)

    def _set_file_attribute(
        self,
        kw, new_value,
        allow_none=False, allow_not_exist=False
    ):
        if not allow_none:
            assert new_value is not None
        elif new_value is not None and not allow_not_exist:
            assert os.path.isfile(new_value)
        self._set_attribute(kw, new_value)

    @property
    def annotation_file(self):
        return self._get_attribute('_annotation_file')

    @annotation_file.setter
    def annotation_file(self, new_value):
        self._set_file_attribute('_annotation_file', new_value, allow_not_exist=True)

    @property
    def mask_file(self):
        return self._get_attribute('_mask_file')

    @mask_file.setter
    def mask_file(self, new_value):
        self._set_file_attribute('_mask_file', new_value, allow_none=True)

    @property
    def keypoint_file(self):
        return self._get_attribute('_keypoint_file')

    @keypoint_file.setter
    def keypoint_file(self, new_value):
        self._set_file_attribute('_keypoint_file', new_value, allow_none=True)

    @property
    def video_file(self):
        return self._get_attribute('_video_file')

    @video_file.setter
    def video_file(self, new_value):
        self._set_file_attribute('_video_file', new_value, allow_none=True)

    @property
    def image_file(self):
        return self._get_attribute('_image_file')

    @image_file.setter
    def image_file(self, new_value):
        self._set_file_attribute('_image_file', new_value, allow_none=True)

    def __init__(
        self,
        annotation_file,
        mask_file=None,
        keypoint_file=None,
        video_file=None,
        image_file=None
    ):

        self.annotation_file = annotation_file
        self.annotation_data = None
        self._images = None
        self._rel_path_lookup = {}

        self.mask_file = mask_file
        self.keypoint_file = keypoint_file
        self.video_file = video_file
        self.image_file = image_file

    def _load_annotation_data(self, raise_error=True):

        try:
            self.annotation_data = sio.loadmat(self.annotation_file)
        except FileNotFoundError:
            if raise_error:
                raise
            else:
                self.annotation_data = None
                return False
        else:
            return True

    def _save_annotation_data(self):

        sio.savemat(
            self.annotation_file,
            self.annotation_data,
            format='5',
            do_compression=True,
            #oned_as={'row' (default), 'column'}
        )

    def verify(self):

        #
        # we assume that we have MatLab files below version 7.3
        #

        self._load_annotation_data()

        #
        # images structured array
        #

        assert 'images' in self.annotation_data, \
            'Require root key missing in annotation file'
        images = self.annotation_data['images']
        assert len(images) == 1
        images = images[0]
        num_images = len(images)

        #
        # required fields of images 
        #

        names = set(images.dtype.names)
        assert names.issuperset((
            'rel_path',
            'mask',
            'bbox',
            'parts'
        )), 'Required keys missing in annotation file'

        #
        # test each required field on 10% of the rows
        #

        for idx in random.sample(range(num_images), int(num_images * 0.1)):

            row = images[idx]

            #
            # rel_path
            #
            
            assert row['rel_path'].dtype.kind in ('S', 'U'), \
                'Datatype for "rel_path" must be Unicode or String'
            rel_path = row['rel_path'].item() 
            assert rel_path is not None and len(rel_path) > 0, \
                'Value missing for "rel_path"'

            #
            # mask
            #

            assert row['mask'].dtype.kind in ('u', 'i', 'b'), \
                'Datatype for "mask" must be unsigned integer, signed integer or boolean'
            assert len(row['mask'].shape) == 2, \
                '"mask" must have two axes'
            assert row['mask'] is not None, \
                'Value missing for "mask"'

            #
            # bbox
            #

            names = set(row['bbox'].dtype.names)
            assert names == set(('x1', 'y1', 'x2', 'y2')), \
                '"bbox" must be an structured array with the fields: x1, y1, x2, y2'
            assert row['bbox'] is not None, \
                'Value missing for "bbox"'

            for kw in names:
                val = row['bbox'][kw]
                assert val.shape == (1, 1), \
                    'Shape of "bbox.{kw}" expected to be: (1, 1)'.format(
                        kw=kw
                    )

                val = val.item()
                assert val.dtype.kind in ('u', 'i', 'f'), \
                    'Datatype for "bbox.{kw}" must be unsigned integer, signed integer or floating-point'.format(
                        kw=kw
                    )
                assert val is not None, \
                    'Value missing for "bbox.{kw}"'.format(kw=kw)

            #
            # parts
            #

            assert row['parts'].shape[0] == 3, \
                    'Shape for "parts" must be: (3, x) where x equals the number of keypoints'
            assert row['parts'] is not None, \
                'Value missing for "parts"'

        print('No issues found: {}'.format(self.annotation_file))

    def _load_annotation_data_structure(self):

        dtype = self.STRUCTURED_DTYPES['images']
        self._images = DynamicRecArray(dtype, size=100)

        # we intentionally exclude everything not required for CMR to run
        if self.annotation_data is not None:
            images = self.annotation_data['images'][0]
            for record in images:
                row = np.array(
                    [tuple(
                        record[name]
                        for name in dtype.names
                    )],
                    dtype=dtype
                )
                self._add_to_images(row)

    def _delete_from_lookup(self, rel_path):

        self._rel_path_lookup[rel_path] = []

    def _find_in_lookup(self, rel_path):

        return self._rel_path_lookup.get(rel_path, [])

    def _delete_from_images(self, rel_path):

        for index in self._find_in_lookup(rel_path):
            self._images.delete(index)

    def _add_to_lookup(self, rel_path, index):

        self._rel_path_lookup = self._rel_path_lookup or {}

        rel_path_indices = self._rel_path_lookup.setdefault(rel_path, [])
        rel_path_indices.append(index)

    def _add_to_images(self, row):

        self._images.extend(row)
        self._add_to_lookup(row['rel_path'][0].item(), self._images.length - 1)

    def _save_annotation_data_structure(self):

        self.annotation_data = {
            'images': self._images.data
        }

    def update(self):

        assert self.video_file is not None or self.image_file is not None, \
            'Video or Image file must be provided'
        assert self.mask_file is not None, \
            'Mask file must be provided'
        assert self.annotation_file is not None, \
            'Annotation file must be provided'

        self._load_annotation_data(raise_error=False)
        self._load_annotation_data_structure()

        the_file = self.video_file or self.image_file

        parts = self._load_keypoint_data()
        masks_boxes = self._load_masks_boxes()

        # delete any reference to the_file
        self._delete_from_images(the_file)

        # add rows to self._images
        dtype = self.STRUCTURED_DTYPES['images']
        for frame_num, mask_box in masks_boxes.items():

            if frame_num not in parts:
                continue

            row = np.array(
                [(
                    np.array([the_file]),
                    mask_box['box'],
                    parts[frame_num],
                    mask_box['mask'],
                )],
                dtype=dtype
            )
            self._add_to_images(row)

        self._save_annotation_data_structure()
        self._save_annotation_data()

    def _load_masks_boxes(self):

        masks_boxes = {}
        with closing(tb.open_file(self.mask_file, mode='r')) as mask_file:

            data_group = mask_file.root.data
            table = data_group.table

            dtype = self.STRUCTURED_DTYPES['bbox']
            for row in table.iterrows():
                box = row['box']
                box[0:2] = np.floor(box[0:2])
                box[2:4] = np.ceil(box[2:4])
                masks_boxes[row['frame_num']] = {
                    'mask': row['mask'].astype(np.bool_),
                    'box': np.array( # WTF
                        [[
                            tuple(
                                np.array([[v]], dtype=np.uint32)
                                for v in box
                            )
                        ]],
                        dtype=dtype
                    ),
                }

        return masks_boxes

    def _load_keypoint_data(self):

        #
        # written for DeepLabCut pandas dataframes
        #

        df = pd.read_hdf(self.keypoint_file)
        num_frames = len(df.index)

        scorer = set(df.columns.get_level_values(0))
        assert len(scorer) == 1
        scorer = scorer.pop()

        keypoints = []
        for kp in df.columns.get_level_values(1):
            if kp not in keypoints:
                keypoints.append(kp)
        #print('keypoints:', keypoints)

        # x, y, likelihood
        parameters = set(df.columns.get_level_values(2))
        assert parameters == set(('x', 'y', 'likelihood'))

        kp_data = []
        for kp in keypoints:
            X = np.around(df[scorer][kp]['x'].values).astype(np.uint16)
            Y = np.around(df[scorer][kp]['y'].values).astype(np.uint16)
            P = df[scorer][kp]['likelihood'].values >= 0.99

            kp_data.append(np.array([X, Y, P]))

        return np.array(kp_data).transpose()

@click.command()
@click.argument('action', type=click.Choice(('update', 'verify')))
@click.argument('annotation', type=click.Path())
@click.option('--mask', '-m', type=click.Path(), help='Mask file of the Video or Image file')
@click.option('--keypoint', '-k', type=click.Path(), help='Keypoint file of the Video or Image file')
@click.option('--video', '-v', type=click.Path(), help='Video file to be added/updated. Frames will be extracted and placed next to the video file')
@click.option('--image', '-i', type=click.Path(), help='Image file to be added/updated')
def do_it(action, annotation, mask, keypoint, video, image):
    '''
    Utility to update and verify the provided <ANNOTATION> MatLab file for
    consumption by CMR
    '''

    assert (
        (action == 'update') and (
            (video is None and image is not None) or
            (video is not None and image is None)
        )
    ), '--video and --image are mutually exclusive. Only one can be provided at one time'

    mgr = AnnotationManager(
        annotation_file=annotation,
        mask_file=mask,
        keypoint_file=keypoint,
        video_file=video,
        image_file=image
    )

    fn = getattr(mgr, action)
    fn()

if __name__ == '__main__':
    do_it()
