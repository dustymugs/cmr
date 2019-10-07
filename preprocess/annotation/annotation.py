import click
import colorsys
from contextlib import closing
import cv2
import copy
import json
import math
import matplotlib as mpl
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.widgets
import mpl_toolkits.axes_grid1
import numpy as np
import os
import os.path as osp
import pandas as pd
from PIL import Image
import random
import re
import scipy.io as sio
import tables as tb
import time

# https://stackoverflow.com/a/41731455/3121505
class PageSlider(matplotlib.widgets.Slider):

    def __init__(self, ax, label, numpages = 10, valinit=0, valfmt='%1d',
                 closedmin=True, closedmax=True,
                 dragging=True, **kwargs):

        self.facecolor=kwargs.get('facecolor',"w")
        self.activecolor = kwargs.pop('activecolor',"b")
        self.fontsize = kwargs.pop('fontsize', 10)
        self.numpages = numpages

        super(PageSlider, self).__init__(ax, label, 0, numpages,
                            valinit=valinit, valfmt=valfmt, **kwargs)

        self.poly.set_visible(False)
        self.vline.set_visible(False)
        self.pageRects = []
        for i in range(numpages):
            facecolor = self.activecolor if i==valinit else self.facecolor
            r  = matplotlib.patches.Rectangle(
                (float(i) / numpages, 0),
                1. / numpages,
                1,
                transform=ax.transAxes,
                facecolor=facecolor
            )
            ax.add_artist(r)
            self.pageRects.append(r)
            ax.text(
                float(i) / numpages + 0.5 / numpages,
                0.5,
                str(i+1),
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=self.fontsize
            )
        self.valtext.set_visible(False)

        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        fax = divider.append_axes("right", size="5%", pad=0.05)
        self.button_back = matplotlib.widgets.Button(
            bax,
            label='\u25C0',
            color=self.facecolor,
            hovercolor=self.activecolor
        )
        self.button_forward = matplotlib.widgets.Button(
            fax,
            label='\u25B6',
            color=self.facecolor,
            hovercolor=self.activecolor
        )
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.backward)
        self.button_forward.on_clicked(self.forward)

    def _update(self, event):
        super(PageSlider, self)._update(event)
        i = int(self.val)
        if i >=self.valmax:
            return
        self._colorize(i)

    def _colorize(self, i):
        for j in range(self.numpages):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def forward(self, event):
        current_i = int(self.val)
        i = current_i+1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)

    def backward(self, event):
        current_i = int(self.val)
        i = current_i-1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)

class DynamicRecArray(object):
    def __init__(self, dtype, size=10):
        self.dtype = np.dtype(dtype) if not isinstance(dtype, np.dtype) else dtype
        self.length = 0
        self.size = size
        self._data = np.empty(self.size, dtype=self.dtype)
        self._is_valid = np.zeros(self.size, dtype=bool)

    def __len__(self):
        return self.length

    def _resize(self):

        if self.length == self.size:

            new_size = int(1.5 * self.size)
            delta = new_size - self.size
            self.size = new_size

            self._data.resize(self.size)
            self._is_valid.resize(self.size)

    def append(self, rec):
        self._resize()
        self._data[self.length] = rec
        self._is_valid[self.length] = True
        self.length += 1

    def extend(self, recs):
        for rec in recs:
            self.append(rec)

    def delete(self, index):

        self._is_valid[index] = False
        # TODO: delete the actual object?

    @property
    def data(self):
        return (self._data[self._is_valid])[:self.length]

class AnnotationManager(object):

    FRAME_FILE_NAME = 'frame_{0:05d}.jpg'
    FRAME_FILE_REGEX = '.*/{}/frame_\d{{5}}\.jpg$'

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

    def _set_path_attribute(
        self,
        kw, new_value,
        type_='file',
        allow_none=False, allow_not_exist=False
    ):
        if not allow_none:
            assert new_value is not None
        if new_value is not None and not allow_not_exist:
            is_type = getattr(osp, 'is{}'.format(type_))
            assert is_type(new_value), \
                'Not a {}: {}'.format(type_, new_value)

        self._set_attribute(kw, new_value)

    @property
    def annotation_file(self):
        return self._get_attribute('_annotation_file')

    @annotation_file.setter
    def annotation_file(self, new_value):
        self._set_path_attribute('_annotation_file', new_value, allow_not_exist=True)

    @property
    def mask_file(self):
        return self._get_attribute('_mask_file')

    @mask_file.setter
    def mask_file(self, new_value):
        self._set_path_attribute('_mask_file', new_value, allow_none=True)

    @property
    def keypoint_file(self):
        return self._get_attribute('_keypoint_file')

    @keypoint_file.setter
    def keypoint_file(self, new_value):
        self._set_path_attribute('_keypoint_file', new_value, allow_none=True)

    @property
    def video_file(self):
        return self._get_attribute('_video_file')

    @video_file.setter
    def video_file(self, new_value):
        self._set_path_attribute('_video_file', new_value, allow_none=True)

    @property
    def image_file(self):
        return self._get_attribute('_image_file')

    @image_file.setter
    def image_file(self, new_value):
        self._set_path_attribute('_image_file', new_value, allow_none=True)

    @property
    def frame_dir(self):
        return self._get_attribute('_frame_dir')

    @frame_dir.setter
    def frame_dir(self, new_value):
        self._set_path_attribute('_frame_dir', new_value, type_='dir', allow_none=True)

    @property
    def ignore_file(self):
        return self._get_attribute('_ignore_file')

    @ignore_file.setter
    def ignore_file(self, new_value):
        self._set_path_attribute('_ignore_file', new_value, allow_none=True)

    @property
    def skeleton_file(self):
        return self._get_attribute('_skeleton_file')

    @skeleton_file.setter
    def skeleton_file(self, new_value):
        self._set_path_attribute('_skeleton_file', new_value, allow_none=True)

    @property
    def max_kp_diff(self):
        return self._max_kp_diff

    @max_kp_diff.setter
    def max_kp_diff(self, new_value):

        if new_value is not None:
            assert new_value >= 0., \
                'max_kp_diff must be a positive value'

        self._max_kp_diff = new_value

    @property
    def exclude_kp(self):
        return self._exclude_kp

    @exclude_kp.setter
    def exclude_kp(self, new_value):

        if new_value is None:
            new_value = tuple()
        assert hasattr(new_value, '__iter__'), \
            '"exclude_kp" is not iterable (e.g. list, set, tuple)'

        self._exclude_kp = new_value

    @property
    def kp_colors(self):
        return self._kp_colors

    @kp_colors.setter
    def kp_colors(self, new_value):
        if new_value is None:
            new_value = tuple()

        assert hasattr(new_value, '__iter__'), \
            '"kp_colors" is not iterable (e.g. list, set, tuple)'

        as_dict = {
            kp: (int(r) / 255., int(g) / 255., int(b) / 255.)
            for kp, r, g, b in [
                v.split(',')
                for v in new_value
            ]
        }

        self._kp_colors = as_dict

    def __init__(
        self,
        annotation_file,
        mask_file=None,
        keypoint_file=None,
        video_file=None,
        image_file=None,
        frame_dir=None,
        no_frames=False,
        ignore_file=None,
        max_kp_diff=None,
        skeleton_file=None,
        exclude_keypoints=None,
        kp_colors=None
    ):

        self.annotation_file = annotation_file
        self.annotation_data = None
        self._images = None
        self._rel_path_lookup = {}

        self.mask_file = mask_file
        self.keypoint_file = keypoint_file
        self.video_file = video_file
        self.image_file = image_file
        self.frame_dir = frame_dir
        self.no_frames = no_frames
        self.ignore_file = ignore_file

        self.max_kp_diff = max_kp_diff
        self.skeleton_file = skeleton_file
        self.exclude_keypoints = exclude_keypoints
        self.kp_colors = kp_colors

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

            if 'keypoints' in self.annotation_data:
                self.keypoints = [
                    kp.strip()
                    for kp in self.annotation_data['keypoints']
                ]
            else:
                self.keypoints = []

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
            assert row['parts'].dtype.kind == 'f', \
                'Datatype for "parts" must be floating point'

        print('No issues found: {}'.format(self.annotation_file))

    def _load_annotation_data_structure(self):

        dtype = self.STRUCTURED_DTYPES['images']
        self._images = DynamicRecArray(dtype, size=100)

        # we intentionally exclude everything not required for CMR to run
        if self.annotation_data is not None:

            try:
                images = self.annotation_data['images'][0]
            except Exception as e:
                pass
            else:
                for record in images:
                    row = np.array(
                        [tuple(
                            record[name]
                            for name in dtype.names
                        )],
                        dtype=dtype
                    )
                    self._add_to_images(row)

                assert self._images.length == images.shape[0]
                assert len(self._rel_path_lookup.keys()) >= images.shape[0]

    def _delete_from_lookup(self, rel_path):

        self._rel_path_lookup[rel_path] = []

    def _find_in_lookup(self, rel_path, regex_pattern=None):

        if regex_pattern is None:

            return self._rel_path_lookup.get(rel_path, [])

        else:

            return {
                k: v
                for k, v in self._rel_path_lookup.items()
                if re.match(regex_pattern, k)
            }

    def _delete_from_images(self, rel_path):

        # videos require deleting anything in the directory of same name
        if rel_path == self.video_file:

            dir_name = osp.splitext(osp.basename(rel_path))[0]

            # delete paths that end with: .../<dir_name>/frame_XXXXX.png
            reo = re.compile(self.FRAME_FILE_REGEX.format(dir_name))

            for frame_path, indexes in self._find_in_lookup(rel_path, reo).items():
                for index in indexes:
                    self._images.delete(index)
                self._delete_from_lookup(frame_path)

        else:

            for index in self._find_in_lookup(rel_path):
                self._images.delete(index)
            self._delete_from_lookup(rel_path)

    def _add_to_lookup(self, rel_path, index):

        self._rel_path_lookup = self._rel_path_lookup or {}

        rel_path_indices = self._rel_path_lookup.setdefault(rel_path, [])
        rel_path_indices.append(index)

    def _add_to_images(self, row):

        self._images.extend(row)
        self._add_to_lookup(row['rel_path'][0].item(), self._images.length - 1)

    def _save_annotation_data_structure(self):

        self.annotation_data = {
            'images': self._images.data,
            'keypoints': self.keypoints
        }

    def _extract_frames(self):

        assert self.video_file is not None
        assert self.frame_dir is not None

        frames_path = osp.abspath(
            osp.join(
                self.frame_dir,
                osp.splitext(osp.basename(self.video_file))[0]
            )
        )

        if self.no_frames:
            return frames_path

        os.makedirs(frames_path, exist_ok=True)

        src = cv2.VideoCapture(self.video_file)
        width = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame = 0
        success, image = src.read()
        while success:

            frame_path = osp.join(
                frames_path,
                self.FRAME_FILE_NAME.format(frame)
            )

            # OpenCV returns images as BGR, convert to RGB
            im = image[..., ::-1]
            im = Image.fromarray(im)

            im.save(frame_path, optimize=True)

            success, image = src.read()
            frame += 1

        return frames_path

    def _in_ignore_file(self, frame_num):

        if not hasattr(self, '_ignore_file_list'):

            self._ignore_file_list = []
            if self.ignore_file is not None:

                with open(self.ignore_file, 'rb') as fh:
                    items = [
                        l.decode()
                        for l in fh.read().splitlines()
                    ]

                    for item in items:

                        if len(item.strip()) < 1:
                            continue

                        parts = item.split('-')
                        num_parts = len(parts)
                        if num_parts == 1:
                            self._ignore_file_list.append(int(parts[0]))
                        elif num_parts == 2:
                            self._ignore_file_list.extend(
                                range(int(parts[0]), int(parts[-1]) + 1)
                            )

        return frame_num in self._ignore_file_list

    def _ignore_frame(self, frame_num, frame_path):
        '''
        Return True to ignore frame
        '''

        if self._in_ignore_file(frame_num):
            return True

        im = Image.open(frame_path)

        # rescale for speed
        im.thumbnail((128, 128))
        ima = np.asarray(im)

        # reorg
        nrgb = ima / 255.
        nhsv = mpl.colors.rgb_to_hsv(nrgb)
        nhsv = np.moveaxis(nhsv, -1, 0)

        # too dark
        values = nhsv[2] # v of hsv
        if np.mean(values) + np.std(values) < 0.5:
            return True

        return False

    def _get_dimensions(self):

        if self.video_file:

            cap = cv2.VideoCapture(self.video_file)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        else:

            im = Image.open(self.image_file)
            width = im.width
            height = im.height

        assert width > 0
        assert height > 0

        return height, width

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
        the_full_file = osp.abspath(the_file)

        # delete any reference to the_file
        self._delete_from_images(the_file)

        height, width = self._get_dimensions()

        parts = self._load_keypoint_data(
            height=height,
            width=width
        )
        masks_boxes = self._load_masks_boxes()

        # if video, extract frames
        frames_path = None
        if (the_file == self.video_file):
            frames_path = self._extract_frames()

        # add rows to self._images
        dtype = self.STRUCTURED_DTYPES['images']
        num_keypoints = len(self.keypoints)
        num_included = 0
        start_num_rows = self._images.data.shape[0]
        for frame_num, mask_box in masks_boxes.items():

            frame_parts = parts[frame_num]
            # make sure most keypoints are usable
            #if np.count_nonzero(frame_parts[2, :]) / num_keypoints < 0.7:
            #    continue

            if frames_path:
                rel_path = osp.join(
                    frames_path,
                    self.FRAME_FILE_NAME.format(frame_num)
                )
            else:
                rel_path = the_full_file

            if self._ignore_frame(frame_num, rel_path):
                continue

            row = np.array(
                [(
                    np.array([rel_path]), # we store absolute paths
                    mask_box['box'],
                    frame_parts,
                    mask_box['mask'],
                )],
                dtype=dtype
            )
            self._add_to_images(row)
            num_included += 1

        assert num_included == (self._images.data.shape[0] - start_num_rows), \
            'Mismatch detected in number of data to be written. Expected {}. Got {}'.format(
                num_included,
                self._images.data.shape[0]
            )

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

    def _filter_keypoints_using_skeleton(self, kp_data):

        # TODO: remove once this works
        return kp_data

        if not self.skeleton_file:
            return kp_data

        with open(self.skeleton_file, 'r') as fh:
            skeleton = json.load(fh)['skeleton']

        keypoint_pairs = [
            [self.keypoints.index(a), self.keypoints.index(b)]
            for (a, b) in skeleton
        ]

        for (a_idx, b_idx) in keypoint_pairs:

            a = kp_data[a_idx]
            b = kp_data[b_idx]

            ax = a[0]
            ay = a[1]
            bx = b[0]
            by = b[1]

            c_length = np.sqrt(np.square(bx - ax) + np.square(by - ay))

            # remove outlier

            # by standard deviation?
            outliers = c_length > c_length.mean() + c_length.std() * 2

            # by acceleration?
            # prepend by two
            _c_length = c_length.copy()
            np.insert(_c_length, 0, _c_length[1:3])
            accel = np.absolute(np.diff(_c_length, n=2))
            outliers = accel > accel.mean() + accel.std() * 2

        return kp_data

    def _load_keypoint_data(self, height=None, width=None):

        #
        # written for DeepLabCut pandas dataframes
        #

        df = pd.read_hdf(self.keypoint_file)
        num_frames = len(df.index)

        scorer = set(df.columns.get_level_values(0))
        assert len(scorer) == 1
        scorer = scorer.pop()

        self.keypoints = []
        for kp in df.columns.get_level_values(1):
            if kp not in self.keypoints:
                self.keypoints.append(kp)

        self._filtered_keypoints()

        # x, y, likelihood
        parameters = set(df.columns.get_level_values(2))
        assert parameters == set(('x', 'y', 'likelihood'))

        kp_data = []
        for kp in self.keypoints:

            X = df[scorer][kp]['x'].values
            Y = df[scorer][kp]['y'].values

            X[X < 0.] = 0.
            Y[Y < 0.] = 0.

            if height is not None:
                Y[Y > height] = height
            if width is not None:
                X[X > width] = width

            # present/absent
            P = (df[scorer][kp]['likelihood'].values >= 0.99)

            # remove outlier keypoints
            if self.max_kp_diff is not None:

                outlier_x = np.absolute(np.diff(X)) > self.max_kp_diff
                outlier_y = np.absolute(np.diff(Y)) > self.max_kp_diff

                outlier_x = np.insert(outlier_x, 0, False)
                outlier_y = np.insert(outlier_y, 0, False)

                if outlier_x.any():
                    P[outlier_x, ...] = False
                if outlier_y.any():
                    P[outlier_y, ...] = False

            kp_data.append(np.array([X, Y, P]))

        kp_data = self._filter_keypoints_using_skeleton(kp_data)

        return np.array(kp_data).transpose()

    def _filtered_keypoints(self):

        self.keypoints = [
            kp
            for kp in self.keypoints
            if (
                kp not in self.exclude_keypoints and
                int(self.keypoints.index(kp)) not in self.exclude_keypoints
            )
        ]

    def _random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        random.seed(0)
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def _show_review_window(
        self,
        images,
        num_keypoints,
        colors,
        num_images_per_page=25
    ):

        plt.close('all')

        fig, ax = plt.subplots(figsize=(14, 9))
        ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.04])
        slider = PageSlider(
            ax_slider,
            'Images',
            num_images_per_page,
            activecolor='orange',
            valstep=1,
        )

        num_images = len(images)
        slider.page_idx = 0
        total_pages = math.ceil(num_images / num_images_per_page)
        slider.start_idx = slider.page_idx * num_images_per_page

        def onkeyrelease(evt):

            if evt.key == 'right':

                if slider.val + 1 == num_images_per_page:

                    next_page_idx = slider.page_idx + 1
                    if next_page_idx < total_pages:
                        i = 0
                        slider.set_val(i)
                        slider._colorize(i)
                        slider.page_idx = next_page_idx
                        update_plot(slider.val)

                else:

                    slider.forward(None)

            elif evt.key == 'left':

                if slider.val - 1 < 0:

                    next_page_idx = slider.page_idx - 1
                    if next_page_idx >= 0:
                        i = num_images_per_page - 1
                        slider.set_val(i)
                        slider._colorize(i)
                        slider.page_idx = next_page_idx
                        update_plot(slider.val)

                else:

                    slider.backward(None)

            elif evt.key == 'up':

                next_page_idx = slider.page_idx + 1
                if next_page_idx < total_pages:
                    slider.page_idx = next_page_idx
                else:
                    i = num_images_per_page - 1
                    slider.set_val(i)
                    slider._colorize(i)

                update_plot(slider.val)

            elif evt.key == 'down':

                next_page_idx = slider.page_idx - 1
                if next_page_idx >= 0:
                    slider.page_idx = next_page_idx
                else:
                    i = 0
                    slider.set_val(i)
                    slider._colorize(i)

                update_plot(slider.val)

        def update_plot(i):

            ax.clear()

            idx = (slider.page_idx * num_images_per_page) + int(i)
            if idx >= num_images or idx < 0:
                return
            row = images[idx]

            rel_path = row['rel_path'].item()
            mask = row['mask']
            bbox = row['bbox'] # x1, y1, x2, y2
            parts = row['parts']

            title = 'Page {current_page}/{total_pages}\n{row_index}/{total_rows} {path}'.format(
                current_page=slider.page_idx + 1,
                total_pages=total_pages,
                row_index=idx,
                total_rows=num_images,
                path=rel_path
            )
            ax.set_title(title)

            for kpi in range(num_keypoints):
                x = parts[0][kpi]
                y = parts[1][kpi]
                p = parts[2][kpi]

                if p != 1.:
                    continue

                label = (
                    self.keypoints[kpi]
                    if len(self.keypoints)
                    else kpi
                )

                ax.scatter(
                    x,
                    y,
                    s=10,
                    c=[colors[kpi]],
                    label=label,
                )
                ax.annotate(
                    label,
                    (x, y),
                    xytext=(x + 10, y + 10),
                    backgroundcolor='black',
                    color='white',
                    alpha=0.5,
                )

            im = Image.open(rel_path)
            ima = np.array(im)
            ax.imshow(
                ima,
                zorder=0,
                extent=[0, im.width, im.height, 0]
            )

            ax.legend(bbox_to_anchor=(1.12, 1.))
            fig.show()

        slider.on_changed(update_plot)
        fig.canvas.mpl_connect('key_release_event', onkeyrelease)

        update_plot(0)
        plt.show()

    def review(self):

        self._load_annotation_data()

        images = self.annotation_data['images'][0]

        num_keypoints = images[0]['parts'].shape[-1]
        colors = self._random_colors(num_keypoints)
        colors = self._merge_colors(colors)

        self._show_review_window(images, num_keypoints, colors)
        plt.close('all')

    def _merge_colors(self, colors):

        assert len(self.keypoints)

        for kp_name, rgb in self.kp_colors.items():

            if kp_name not in self.keypoints:
                continue

            idx = self.keypoints.index(kp_name)
            colors[idx] = rgb

        return colors

@click.command()
@click.argument('action', type=click.Choice(('update', 'verify', 'review')))
@click.argument('annotation', type=click.Path())
@click.option('--mask', '-m', type=click.Path(), help='Mask file of the Video or Image file')
@click.option('--keypoint', '-k', type=click.Path(), help='Keypoint file of the Video or Image file')
@click.option('--video', '-v', type=click.Path(), help='Video file to be added/updated. Frames will be extracted and placed next to the video file')
@click.option('--image', '-i', type=click.Path(), help='Image file to be added/updated')
@click.option('--frame-dir', type=click.Path(), help='Existing directory in which vidoe frames will be placed')
@click.option('--no-frames', is_flag=True, default=False, help='Do not extract frames from videos')
@click.option('--ignore', type=click.Path(), help='Ignore file with list of frame numbers (0-based) to exclude')
@click.option('--max-kp-diff', type=float, help='Maximum pixel difference for a keypoint from frame to frame')
@click.option('--skeleton', type=click.Path(), help='Skeleton file in JSON format')
@click.option('--exclude-kp', multiple=True, help='Name or 0-based Index of Keypoint to exclude')
@click.option('--kp-color', multiple=True, help='Comma separated pair of values (Keypoint, RGB). e.g. --kp-color=Nose,255,0,0')
def do_it(
    action,
    annotation,
    mask,
    keypoint,
    video,
    image,
    frame_dir,
    no_frames,
    ignore,
    max_kp_diff,
    skeleton,
    exclude_kp,
    kp_color,
):
    '''
    Utility to update, verify, review the provided <ANNOTATION> MatLab file for
    consumption by CMR
    '''

    if action == 'update':
        assert (
            (video is None and image is not None) or
            (video is not None and image is None)
        ), '--video and --image are mutually exclusive. Only one can be provided at one time'
        if video is not None:
            assert frame_dir is not None, \
                '--frame-dir is required if --video is provided'

    mgr = AnnotationManager(
        annotation_file=annotation,
        mask_file=mask,
        keypoint_file=keypoint,
        video_file=video,
        image_file=image,
        frame_dir=frame_dir,
        no_frames=no_frames,
        ignore_file=ignore,
        max_kp_diff=max_kp_diff,
        skeleton_file=skeleton,
        exclude_keypoints=exclude_kp,
        kp_colors=kp_color
    )

    fn = getattr(mgr, action)
    fn()

if __name__ == '__main__':
    do_it()
