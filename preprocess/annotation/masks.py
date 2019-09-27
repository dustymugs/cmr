import click
import colorsys
import cv2
from contextlib import closing
from itertools import zip_longest
import math
import numpy as np
import os
import random
import skimage
from skimage.measure import find_contours
import tables as tb
import torch
import torchvision

import matplotlib as mpl
import matplotlib.pyplot as plt

class SegmentImage(object):

    COCO_IDS = (
        21, # cow
    )

    VIDEO_EXTENSIONS = (
        'avi',
        'mp4',
        'mkv',
    )
    IMAGE_EXTENSIONS = (
        'jpg',
        'jpeg',
        'png',
    )

    @property
    def model(self):

        if not hasattr(self, '_model'):

            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            model.cuda()

            self._model = model

        return self._model

    def image_to_tensor(self, image_path=None, image_data=None, cuda=True):

        if image_data is not None:
            image = image_data
        else:
            image = skimage.io.imread(image_path)

        image = np.moveaxis(image, -1, 0) # from H,W,C to C,H,W
        image = image.astype(np.float32)
        image *= 1./255.

        tensor = torch.from_numpy(image)
        return tensor if not cuda else tensor.cuda()

    def random_colors(self, N, bright=True):
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

    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(
                mask == 1,
                image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                image[:, :, c]
            )

        return image

    def _filter_instances(
        self,
        class_ids,
        boxes,
        width,
        height,
        scores=None,
        min_box_area_ratio=0.2
    ):

        max_area = width * height * 1.

        if self.only_class_ids:

            filtered_instances = []

            if scores is None:
                scores = []

            for idx, (class_id, box, score) in \
                enumerate(zip_longest(class_ids, boxes, scores, fillvalue=0.)):

                x1, y1, x2, y2 = box
                box_area = math.fabs(x2 - x1) * math.fabs(y2 - y1) * 1.

                if (
                    class_id in self.only_class_ids and
                    score > self.min_score and
                    box_area / max_area >= min_box_area_ratio
                ):

                    filtered_instances.append(idx)

        else:

            filtered_instances = list(range(boxes.shape[0]))

        return filtered_instances

    def overlay_instances(
        self,
        image,
        boxes,
        masks,
        class_ids,
        class_names=None,
        scores=None,
        title="",
        show_mask=True, show_bbox=True,
        colors=None,
        captions=None,
    ):

        # Number of instances
        N = len(self.filtered_instances)
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[0] == class_ids.shape[0]

        dpi = plt.figure().dpi
        height, width = image.shape[:2]
        figsize = (width / dpi, height / dpi)
        fig, ax = plt.subplots(1, figsize=figsize)

        if class_names is None:
            class_names = {
                class_id: str(class_id)
                for class_id in set(self.only_class_ids or class_ids)
            }

        # Generate random colors
        colors = colors or {
            class_id: color
            for class_id, color in zip(
                class_names.values(),
                self.random_colors(len(class_names.keys()))
            )
        }

        #ax.axis('off')

        masked_image = image.astype(np.uint32).copy()
        for i in self.filtered_instances:

            class_id = class_ids[i]
            label = class_names[class_id]
            color = colors[label]
            mask = masks[i][0]

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            x1, y1, x2, y2 = boxes[i]
            if show_bbox:
                p = mpl.patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=1,
                    alpha=0.7,
                    linestyle="dashed",
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(p)

            # Label
            if captions is not False:
                if captions is None:
                    score = scores[i] if scores is not None else None
                    caption = "{} {:.3f}".format(label, score) if score else label
                else:
                    caption = captions[i]
                ax.text(
                    x1,
                    y1 + 8,
                    caption,
                    color='w', size=8, backgroundcolor="k"
                )

            # Mask
            if show_mask:
                masked_image = self.apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = mpl.patches.Polygon(
                    verts,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=1
                )
                ax.add_patch(p)

        fig = ax.imshow(masked_image.astype(np.uint8)).figure
        fig.canvas.draw()

        plt.close(fig=1)
        plt.show(block=True)

    def __init__(
        self,
        videos=None,
        images=None,
        video_dirs=None,
        image_dirs=None,
        only_class_ids=None,
        min_score=0.5,
        min_mask_score=0.5,
        video_extensions=None,
        image_extensions=None,
        visualize=False,
    ):

        self.videos = videos
        self.images = images
        self.video_dirs = video_dirs
        self.image_dirs = image_dirs
        self.only_class_ids = only_class_ids
        self.min_score = min_score
        self.min_mask_score = min_mask_score
        self.video_extensions = video_extensions or self.VIDEO_EXTENSIONS
        self.image_extensions = image_extensions or self.IMAGE_EXTENSIONS
        self.visualize = visualize

        # Generate random colors for COCO
        self.colors = {
            class_id: color
            for class_id, color in zip(
                range(1, 91),
                self.random_colors(90)
            )
        }

    @property
    def videos(self):
        return self._videos

    @videos.setter
    def videos(self, new_videos):
        if new_videos is not None:
            assert all([os.path.isfile(v) for v in new_videos])
        else:
            new_videos = []

        self._videos = new_videos

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, new_images):
        if new_images is not None:
            assert all([os.path.isfile(v) for v in new_images])
        else:
            new_images = []

        self._images = new_images

    @property
    def video_dirs(self):
        return self._video_dirs

    @video_dirs.setter
    def video_dirs(self, new_video_dirs):
        if new_video_dirs is not None:
            assert all([os.path.isdir(v) for v in new_video_dirs])
        else:
            new_video_dirs = []

        self._video_dirs = [
            os.path.abspath(d)
            for d in new_video_dirs
        ]

    @property
    def image_dirs(self):
        return self._image_dirs

    @image_dirs.setter
    def image_dirs(self, new_image_dirs):
        if new_image_dirs is not None:
            assert all([os.path.isdir(v) for v in new_image_dirs])
        else:
            new_image_dirs = []

        self._image_dirs = [
            os.path.abspath(d)
            for d in new_image_dirs
        ]

    @property
    def only_class_ids(self):
        return self._only_class_ids

    @only_class_ids.setter
    def only_class_ids(self, new_class_ids):
        if new_class_ids is not None:
            assert all([
                True if 0 < cid <= 90 else False
                for cid in new_class_ids
            ])
        else:
            new_class_ids = []

        self._only_class_ids = new_class_ids

    @property
    def min_score(self):
        return self._min_score

    @min_score.setter
    def min_score(self, new_min_score):

        new_min_score = new_min_score or 0.
        assert 0. <= new_min_score <= 1.

        self._min_score = new_min_score * 1.

    @property
    def min_mask_score(self):
        return self._min_mask_score

    @min_mask_score.setter
    def min_mask_score(self, new_min_mask_score):

        new_min_mask_score = new_min_mask_score or 0.
        assert 0. <= new_min_mask_score <= 1.

        self._min_mask_score = new_min_mask_score * 1.

    def _write_data(
        self,
        h5_file,
        file_name,
        file_type,
        index,
        class_id,
        score,
        box,
        mask,
        frame_num=0
    ):

        MaskInfo = type(
            'MaskInfo',
            (tb.IsDescription,),
            dict(
                id=tb.UInt32Col(),
                file_name=tb.StringCol(itemsize=8192),
                file_type=tb.StringCol(itemsize=256),
                frame_num=tb.UInt32Col(),
                class_id=tb.UInt32Col(),
                score=tb.Float32Col(),
                box=tb.Float32Col(shape=(4,)), # x1, y1, x2, y2
                mask=tb.BoolCol(shape=mask.shape), # variable
            )
        )

        try:
            data_group = h5_file.root.data
        except tb.exceptions.NoSuchNodeError:
            data_group = h5_file.create_group('/', 'data')

        try:
            table = data_group.table
        except tb.exceptions.NoSuchNodeError:
            table = h5_file.create_table(data_group, 'table', MaskInfo)

        try:
            masks_group = h5_file.root.masks
        except tb.exceptions.NoSuchNodeError:
            masks_group = h5_file.create_group('/', 'masks')

        row = table.row

        row['id'] = table.nrows + 1
        row['file_name'] = file_name
        row['file_type'] = file_type
        row['frame_num'] = frame_num
        row['class_id'] = class_id
        row['score'] = score
        row['box'] = box

        mask_array = h5_file.create_carray(
            masks_group,
            'mask_{}_{}'.format(frame_num, index),
            obj=mask,
        )

        row['mask'] = mask_array
        row.append()

        table.flush()

    def process(self):

        self._process_images()
        self._process_image_dirs()

        self._process_videos()
        self._process_video_dirs()

    def _process_image_frame(self, image_path, frame=None, frame_num=0):

        if frame is not None:
            tensor = self.image_to_tensor(image_data=frame)
        else:
            tensor = self.image_to_tensor(image_path=image_path)

        channels, height, width = tensor.shape
        results = self.model([tensor])[0]

        boxes = results['boxes'].detach().cpu().numpy()
        masks = results['masks'].detach().cpu().numpy() >= self.min_mask_score
        class_ids = results['labels'].detach().cpu().numpy()
        scores = results['scores'].detach().cpu().numpy()

        self.filtered_instances = self._filter_instances(
            class_ids=class_ids,
            boxes=boxes,
            scores=scores,
            width=width,
            height=height,
        )

        if self.visualize:
            self.overlay_instances(
                image=(
                    frame
                    if frame is not None
                    else skimage.io.imread(image_path)
                ),
                boxes=boxes,
                masks=masks,
                class_ids=class_ids,
                scores=scores,
                show_bbox=True,
                show_mask=True,
            )

        h5_name = os.path.splitext(image_path)[0] + '.mask.h5'
        filters = tb.Filters(
            complib='blosc',
            complevel=9,
            bitshuffle=True,
        )

        file_mode = 'w' if frame_num < 1 else 'a'
        with closing(
            tb.open_file(h5_name, mode=file_mode, filters=filters)
        ) as h5_file:

            for i, idx in enumerate(self.filtered_instances):
                box = boxes[idx]
                mask = masks[idx]
                class_id = class_ids[idx]
                score = scores[idx]

                self._write_data(
                    h5_file=h5_file,
                    file_name=image_path,
                    file_type='image' if frame is not None else 'video',
                    frame_num=frame_num,
                    class_id=class_id,
                    index=i,
                    score=score,
                    box=box,
                    mask=mask[0],
                )

    def _process_images(self, images=None):

        image_paths = images or self.images

        for image_path in image_paths:

            print('Processing Image: {}'.format(image_path))
            self._process_image_frame(image_path=image_path)

    def _process_image_dirs(self):

        for image_dir in self.image_dirs:

            for root, dirs, files in os.walk(image_dir):

                for f in files:
                    image_path = os.path.join(root, f)
                    if os.path.splitext(image_path)[-1][1:].lower() not in self.image_extensions:
                        continue

                    try:
                        skimage.io.imread(image_path)
                    except Exception:
                        continue
                    else:
                        self._process_image_frame(image_path=image_path)

    def _process_videos(self, videos=None):

        video_paths = videos or self.videos

        for video_path in video_paths:

            print('Processing Video: {}'.format(video_path))
            src = cv2.VideoCapture(video_path)

            frame_num = 0
            success, frame = src.read()
            while success:

                # OpenCV returns images as BGR, convert to RGB
                frame = frame[..., ::-1]

                self._process_image_frame(
                    image_path=video_path,
                    frame=frame,
                    frame_num=frame_num
                )

                success, frame = src.read()
                frame_num += 1

    def _process_video_dirs(self):

        for video_dir in self.video_dirs:

            for root, dirs, files in os.walk(video_dir):

                video_paths = set()
                for f in files:
                    video_path = os.path.join(root, f)
                    if os.path.splitext(video_path)[-1][1:].lower() not in self.video_extensions:
                        continue

                    try:
                        cv2.VideoCapture(video_path)
                    except Exception:
                        continue
                    else:
                        video_paths.add(video_path)

                self._process_videos(videos=video_paths)

@click.command()
@click.option('--video', multiple=True, type=click.Path(), help='Video for Mask-RCNN to process')
@click.option('--image', multiple=True, type=click.Path(), help='Image for Mask-RCNN to process')
@click.option('--video-dir', multiple=True, type=click.Path(), help='Directory of videos for Mask-RCNN to process')
@click.option('--image-dir', multiple=True, type=click.Path(), help='Directory of images for Mask-RCNN to process')
@click.option('--class-id', type=int, multiple=True, default=SegmentImage.COCO_IDS, help='COCO ID to filter Mask-RCNN results')
@click.option('--min-score', type=float, default=0.5, help='Mask-RCNN scores below this value will be excluded from the resultset')
@click.option('--video-extension', '-vx', multiple=True)
@click.option('--image-extension', '-ix', multiple=True)
@click.option('--visualize', '-v', is_flag=True, help='Show Mask-RCNN results')
def do_it(
    video,
    image,
    video_dir,
    image_dir,
    class_id,
    min_score,
    video_extension,
    image_extension,
    visualize
):

    si = SegmentImage(
        videos=video,
        images=image,
        video_dirs=video_dir,
        image_dirs=image_dir,
        only_class_ids=class_id,
        min_score=min_score,
        video_extensions=video_extension,
        image_extensions=image_extension,
        visualize=visualize,
    )

    si.process()

if __name__ == '__main__':
    do_it()
