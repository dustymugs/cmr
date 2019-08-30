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

2. Download our CUB annotation mat files and pre-computed SfM outputs.  For structural details regarding these files, read [Data File Structures](files.md)

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
