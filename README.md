# Purpose

This repo is a fork of the [main project repo](https://github.com/akanazawa/cmr) for CMR. This repo has the following goals:

- [X] Docker image to sandbox the project
- [X] Octave support as alternative to MatLab for preprocessing scripts
- [X] Add and enhance documentation with regards to training and data files
- [X] Updated PyTorch support (1.2+)
- [X] Updated Python dependencies
- [X] Python3 support
- [ ] Scripts to reproduce the structure of CUB data annotation .mat files (and thus create appropriate files for your own projects)
- [ ] Replace MatLab code with pure Python implementation

# Learning Category-Specific Mesh Reconstruction from Image Collections

Angjoo Kanazawa<sup>\*</sup>, Shubham Tulsiani<sup>\*</sup>, Alexei A. Efros, Jitendra Malik

University of California, Berkeley
In ECCV, 2018

[Project Page](https://akanazawa.github.io/cmr/)
[Paper](https://arxiv.org/pdf/1803.07549.pdf)
![Teaser Image](https://akanazawa.github.io/cmr/resources/images/teaser.png)

### Docker

Use the Docker image provided as it contains:

- All scripts/tooling required to run CMR
- Octave for running the preprocess steps

#### Pull Docker Image

```
docker pull dustymugs/cmr
```

#### Start Docker Container

```
cd /cmr
docker/run_x11.sh --runtime=nvidia -it -v /PATH/TO/cmr:/cmr -p 8888:8888 -p 8097:8097 cmr bash
```

### Demo

1. Download the trained model into the `/cmr` directory. You should see `cmr/cachedir/snapshots/bird_net/` (the original trained weights) and `cmr/cachedir/snapshots/bird2/` (weights trained with this Docker image)

```
cd /cmr
wget https://drive.google.com/open?id=1WVDe3H6QK5K8HCwkKekdEMIUY0Mo0ATD
tar xf model.tar.bz2
```

Expected MD5 hash for model.tar.bz2

```
fe61bb79e5893c0a6a30749cb93307f1
```

2. Run the demo:

Note that due to the module-drive approach, you will need to run the `python` commands from the root path `/`

```
cd /
python -m cmr.demo --name bird_net --num_train_epoch 500 --img_path cmr/demo_data/img1.jpg
python -m cmr.demo --name bird_net --num_train_epoch 500 --img_path cmr/demo_data/img2.jpg
python -m cmr.demo --name bird_net --num_train_epoch 500 --img_path cmr/demo_data/birdie.jpg
```

Substitute `bird2` for `bird_net` to use the weights trained with this Docker image

### Training
Please see [doc/train.md](doc/train.md)

### Citation
If you use this code for your research, please consider citing:
```
@inProceedings{cmrKanazawa18,
  title={Learning Category-Specific Mesh Reconstruction
  from Image Collections},
  author = {Angjoo Kanazawa and
  Shubham Tulsiani
  and Alexei A. Efros
  and Jitendra Malik},
  booktitle={ECCV},
  year={2018}
}
```
