# Data File Structures

## CUB annotation mat files
  * in the CUB annotation mat files, all pixel coordinates are **1-based** (vs the normal 0-based for X and Y axes)
  * All CUB annotation mat files (e.g. `train_cub_cleaned.mat`, `testval_cub_cleaned.mat`) contain only one variable: `images`

### Details of `images` struct array
```
images = 

  1×5747 struct array containing the fields:

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

#### Fields of `images` struct array

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
  * train - boolean (0/1) flag indicating if row is part of training dataset
```
>> images(1).train
ans = 1
```
  * test - boolean (0/1) flag indicating if row is part of training dataset
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
  * parts - locations of the bird's keypoints in pixel coordinates. For each row in array, array index = keypoint id, element 1 = X, element 2 = Y, element 3 = "is present" boolean flag
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
  * mask - segmentation mask of the bird in the image from Mask-RCNN. A boolean (0/1) array of same dimension as image indicating which pixels are masked

```
>> images(1).mask
ans =

 Columns 1 through 63:

  0  0  0  0  0  0  0 ...
  ...

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
#### Required fields of `images` struct array

The following fields are required for CMR training:
* rel_path
* mask
* bbox
* parts

## CUB SfM annotations mat files

  * All CUB sfm mat files (e.g. `anno_train.mat`) contain at least these three variables: `sfm_anno`, `S`, `conv_tri`. These three variables are required for CMR training.

### Details of `sfm_anno` struct array

`sfm_anno` contains information required to transform the mean shape to each bird's unique shape

```
>> sfm_anno
sfm_anno =

  1x5964 struct array containing the fields:

    rot
    scale
    trans

>> whos('sfm_anno')
Variables in the current scope:

   Attr Name          Size                     Bytes  Class
   ==== ====          ====                     =====  =====
        sfm_anno      1x5964                  572544  struct

Total is 5964 elements using 572544 bytes
```

#### Fields of `sfm_anno` struct array

 * rot - rotation
```
>> sfm_anno(1).rot
ans =

  -0.303131  -0.807023   0.506780
  -0.910085   0.087464  -0.405086
   0.282589  -0.584007  -0.760973

>> whos('sfm_anno(1).rot')
Variables in the current scope:

   Attr Name                 Size                     Bytes  Class
   ==== ====                 ====                     =====  =====
        sfm_anno(1).rot      3x3                         72  double
```
 * scale - scale
```
>> sfm_anno(1).scale
ans =  81.948

>> whos('sfm_anno(1).scale')
Variables in the current scope:

   Attr Name                   Size                     Bytes  Class
   ==== ====                   ====                     =====  =====
        sfm_anno(1).scale      1x1                          8  double
```
 * trans - translation
```
>> sfm_anno(1).trans
ans =

   234.27
   151.87

>> whos('sfm_anno(1).trans')
Variables in the current scope:

   Attr Name                   Size                     Bytes  Class
   ==== ====                   ====                     =====  =====
        sfm_anno(1).trans      2x1                         16  double
```

### Details of `S` array

`S` is the mean shape built from the shapes of all training data. The vertices of the mean shape are the mean positions of the  keypoints

```
>> S
S =

 Columns 1 through 14:

   3.6065e-16  -8.9135e-16   2.9800e-16  -5.5840e-17  -3.3744e-16  -6.6210e-16   1.9574e-02   2.7820e-01   3.6425e-01   1.9550e-16  -1.9574e-02  -2.7820e-01  -3.6425e-01   8.5400e-16
   1.3454e-01  -6.4203e-01   1.9811e-01  -1.6509e-01  -4.9102e-01  -5.9724e-01  -4.7580e-01   6.0424e-01   2.8694e-01  -1.8298e-01  -4.7580e-01   6.0424e-01   2.8694e-01   1.3329e+00
   2.6966e-01  -4.6560e-02  -3.6491e-01  -2.7358e-01   2.6940e-01   1.1551e-01   1.3329e-01  -4.6213e-01   1.1462e-01   2.4593e-01   1.3329e-01  -4.6213e-01   1.1462e-01   2.6940e-01

 Column 15:

  -5.0415e-16
  -4.1792e-01
  -5.6380e-02

>> whos('S')
Variables in the current scope:

   Attr Name        Size                     Bytes  Class
   ==== ====        ====                     =====  =====
        S           3x15                       360  double
```

### Details of `conv_tri` array

`conv_tri` is the convex hull of the mean shape `S`

```
>> conv_tri
conv_tri =

    4    2    8
    4   12    2
    1   14    9
    ...
   10    7    9
   10    1    9
   10    1   13

>> whos('conv_tri')
Variables in the current scope:

   Attr Name          Size                     Bytes  Class
   ==== ====          ====                     =====  =====
        conv_tri    144x3                       3456  double
```
