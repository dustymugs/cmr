"""
Script for the bird shape, pose and texture experiment.

The model takes imgs, outputs the deformation to the mesh & camera parameters
Loss consists of:
- keypoint reprojection loss
- mask reprojection loss
- smoothness/laplacian priors on triangles
- texture reprojection losses

example usage : python -m cmr.experiments.bird --name=bird_shape --plot_scalars --save_epoch_freq=1 --batch_size=8 --display_visuals --display_freq=2000
"""

from __future__ import absolute_import

from absl import app, flags
import torch

from .shape import ShapeTrainer
from ..data import bird as bird_data

opts = flags.FLAGS

class BirdTrainer(ShapeTrainer):

    DATA_MODULE = bird_data

def main(_):
    torch.manual_seed(0)
    trainer = BirdTrainer(opts)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
