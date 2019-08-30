## Big questions

* Build the CUB annotation mat files
  * in the CUB annotation mat files, all pixel coordinates are 1-based (vs the normal 0-based for X and Y axes)
  * `train_cub_cleaned.mat` and `testval_cub_cleaned.mat` have same structure and contain only one variable:
```
images = 

  1×5747 struct array with fields:

    id
    rel_path
    train
    test
    bbox
    width
    height
    parts
    mask
    class_id
```

  * id - row identifier (integer autonumber?)
```
>> images(1).id
ans = 2
```
  * rel_path - relative path to image data
```
>> images(1).rel_path
ans = 001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg
```
  * train - boolean flag indicating if row is part of training dataset
```
>> images(1).train
ans = 1
```
  * test - boolean flag indicating if row is part of training dataset
```
>> images(1).test
ans = 0
```
  * bbox - bounding box of the bird in image in pixel coordinates
```
>> images(1).bbox
ans =

  scalar structure containing the fields:

    x1 = 139
    y1 = 30
    x2 = 291
    y2 = 293
```
  * width - width of image. number of pixels along the X axis
```
>> images(1).width
ans =  500
```
  * height - height of image. number of pixels along the Y axis
```
>> images(1).height
ans =  336
```
  * parts - locations of the bird's keypoints/landmarks in pixel coordinates. For each row in array, array index = keypoint id, element 1 = X, element 2 = Y, element 3 = "is present" boolean flag
```
>> images(1).parts
ans =

   228   282     0   248   266   272     0     0   208   256   270     0   234   163   260
   138   154     0   158   141   144     0     0   102   141   146     0   193   155   155
     1     1     0     1     1     1     0     0     1     1     1     0     1     1     1

>> whos('images(1).parts')
Variables in the current scope:

   Attr Name                 Size                     Bytes  Class
   ==== ====                 ====                     =====  =====
        images(1).parts      3x15                       360  double
```
  * mask - segmentation mask of the bird in the image from Mask-RCNN. A boolean array of same dimension as image indicating which pixels are masked

```
>> images(1).mask
ans =

 Columns 1 through 63:

  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 ...

>> whos('images(1).mask')
Variables in the current scope:

   Attr Name                Size                     Bytes  Class
   ==== ====                ====                     =====  =====
        images(1).mask    336x500                   168000  logical
```
  * class_id - the class (bird types) of the bird in the image from the CUB dataset
```
>> images(1).class_id
ans = 1
```

## Pre-reqs

These instructions assume you are using the Docker Image

```
docker/run_x11.sh --runtime=nvidia -it -v /PATH/TO/cmr:/cmr -p 8888:8888 cmr bash
```

### CUB Data

1. Download CUB-200-2011 images somewhere:

```
cd /cmr
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar xf CUB_200_2011.tgz
```

2. Download our CUB annotation mat files and pre-computed SfM outputs.

```
cd /cmr
wget https://people.eecs.berkeley.edu/~shubhtuls/cachedir/cmr/cachedir.tar.gz
tar xf cachedir.tar.gz
```

Expected MD5 hash for cachedir.tar.gz

```
89842f8937a136bfdd7106e80f88d30f
```

#### Computing SfM

We provide the computed SfM. If you want to compute them yourself, run the following. Note that octave is slower

From MatLab or Octave, run:

```
cd /cmr/preprocess/cub
main
```

When prompted for 3d model alignment check, only respond ***y*** when the following is correct:

* legs are negative and wings are positive along Z axis
* beak is negative and tail is positive along Y axis
* right side (e.g RLeg, RWing) is negative and left size (e.g. LLeg, LWing) is positive along X axis

##### Octave

Start Octave in the Docker container:

```
su -c octave - cmr
```

### Model training
Change the `name` to whatever you want to call. Also see `shape.py` to adjust
hyper-parameters (for eg. increase `tex_loss_wt` and `text_dt_loss_wt` if you
want better texture, increase texture resolution with `tex_size`).
See `nnutils/mesh_net.py` and `nnutils/train_utils.py` for more model/training options.

```
cd /
python -m cmr.experiments.shape --name=bird_net --display_port 8087
```

### Evaluation
We provide evaluation code to compute the IOU curves in the paper.
Command below runs the model with different ablation settings.
Run it from one directory above the `cmr` directory.

```
cd /
python -m cmr.benchmark.run_evals --split val  --name bird_net --num_train_epoch 500
```

Then, run 

```
cd /
python -m cmr.benchmark.plot_curvess --split val  --name bird_net --num_train_epoch 500
```
in order to see the IOU curve.
