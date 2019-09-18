# Training the model

## Pre-reqs

These instructions assume you are using the Docker Image

```
docker/run_x11.sh --runtime=nvidia -it -v /PATH/TO/cmr:/cmr -p 8888:8888 -p 8097:8097 cmr bash
```

### CUB Data

1. Download CUB-200-2011 images somewhere:

```
cd /cmr
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar xf CUB_200_2011.tgz
```

2. Download the CUB Annotation and pre-computed SfM files (~30MB).  For structural details regarding these files, read [doc/files.md](files.md). For guidance on building your own Anotation files, read [doc/annotation.md](annotation.md)

```
cd /cmr
wget https://drive.google.com/open?id=1LCq949ppsptaa7Bsp-uYC6B6YF10huZY
tar xf cachedir.tar.xz
```

Expected MD5 hash for cachedir.tar.xz

```
07785fc0027f6370027f0106125350d2
```

#### Computing SfM

We provide the computed SfM. If you want to compute them yourself, run the following. Note that Octave is slower than MatLab but it's free.

From MatLab or Octave, run:

```
cd /cmr/preprocess/cub
main
```

When prompted for 3d model alignment check, only respond ***y*** when the following is correct:

* legs are negative and wings are positive along Z axis
* beak is negative and tail is positive along Y axis
* right side (e.g `RLeg`, `RWing`) is negative and left size (e.g. `LLeg`, `LWing`) is positive along X axis

##### Octave

Start Octave in the Docker container:

```
su -c octave - cmr
```

### Model training

Start the Visdom server

```
python -m visdom.server > /cmr/visdom.log 2>&1 &
```

Change the value of `--name` to whatever you want to call the model and change `--cub_dir` to where you extracted the CUB dataset. See [doc/flags.md](flags.md) for model/training options.

```
cd /
python -m cmr.experiments.shape --name=bird_net --display_visuals --plot_scalars --cub_dir=/cmr/CUB_200_2011/
```

To monitor training progress, use your browser to connect to the Visdom server at port 8097

```
http://localhost:8097
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
