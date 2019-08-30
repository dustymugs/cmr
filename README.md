# Learning Category-Specific Mesh Reconstruction from Image Collections

Angjoo Kanazawa<sup>\*</sup>, Shubham Tulsiani<sup>\*</sup>, Alexei A. Efros, Jitendra Malik

University of California, Berkeley
In ECCV, 2018

[Project Page](https://akanazawa.github.io/cmr/)
![Teaser Image](https://akanazawa.github.io/cmr/resources/images/teaser.png)

### Docker

Use the Docker image provided as it contains:

- All scripts/tooling required to run CMR
- Octave for running the preprocess steps

#### Start Docker Container

```
cd /cmr
docker/run_x11.sh --runtime=nvidia -it -v /PATH/TO/cmr:/cmr -p 8888:8888 -p 8097:8097 cmr bash
```

#### Install CMR Dependencies

Once within the Docker Container, run:

```
init_cmr.sh
```

This will install all the requirements of CMR and external dependencies (e.g. Neural Mesh Renderer and Perceptual loss)

### Demo

1. Download the trained model into the `/cmr` directory. You should see `cmr/cachedir/snapshots/bird_net/`

```
cd /cmr
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/cmr/model.tar.gz
tar xf model.tar.gz
```

Expected MD5 hash for model.tar.gz

```
b21c87ec5dae4414a21086d631afdb30
```

2. Run the demo:

Note that due to the module-drive approach, you will need to run the `python` commands from the root path `/`

```
cd /
python -m cmr.demo --name bird_net --num_train_epoch 500 --img_path cmr/demo_data/img1.jpg
python -m cmr.demo --name bird_net --num_train_epoch 500 --img_path cmr/demo_data/birdie.jpg
```

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
