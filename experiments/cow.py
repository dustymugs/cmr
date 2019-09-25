"""
Script for the cow shape, pose and texture experiment.

The model takes imgs, outputs the deformation to the mesh & camera parameters
Loss consists of:
- keypoint reprojection loss
- mask reprojection loss
- smoothness/laplacian priors on triangles
- texture reprojection losses

example usage : python -m cmr.experiments.cow --name=cow --plot_scalars --save_epoch_freq=100 --batch_size=8 --display_visuals --display_freq=2000
"""

from __future__ import absolute_import

from absl import app, flags
import torch

from .shape import ShapeTrainer
from ..data import cow as cow_data

opts = flags.FLAGS

class CowTrainer(ShapeTrainer):

    DATA_MODULE = cow_data

def main(_):
    torch.manual_seed(0)
    trainer = CowTrainer(opts)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
