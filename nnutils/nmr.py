from __future__ import absolute_import

import neural_renderer as nr
import torch

from . import geom_utils

__all__ = [
    'NeuralRenderer'
]

########################################################################
############## Wrapper torch module for Neural Renderer ################
########################################################################
class NeuralRenderer(torch.nn.Module):

    def __init__(self, img_size=256, cuda_device=None):
        super(NeuralRenderer, self).__init__()
        self.cuda_device = cuda_device

        self.renderer = nr.Renderer(
            camera_mode='look_at'
        )

        # Adjust the core renderer
        self.renderer.image_size = img_size
        self.renderer.perspective = False

        # Set a default camera to be at (0, 0, -2.732)
        self.renderer.eye = [0, 0, -2.732]

        # Make it a bit brighter for vis
        self.renderer.light_intensity_ambient = 0.8

        self.proj_fn = geom_utils.orthographic_proj_withz
        self.offset_z = 5.

    def ambient_light_only(self):
        # Make light only ambient.
        self.renderer.light_intensity_ambient = 1
        self.renderer.light_intensity_directional = 0

    def set_bgcolor(self, color):
        self.renderer.background_color = color

    def project_points(self, verts, cams):
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]

    def _new_tensor(self, tensor, requires_grad=True):
        # intentionally detach from current graph

        new_tensor = tensor.clone().detach().requires_grad_()

        if tensor.device is not None or self.cuda_device is not None:
            new_tensor = new_tensor.cuda(
                device=self.cuda_device or tensor.device
            )
        else:
            new_tensor = new_tensor.cpu()

        return new_tensor

    def forward(self, vertices, faces, cams, textures=None):

        v = self.proj_fn(vertices, cams, offset_z=self.offset_z)
        f = faces.to(device=v.device, dtype=torch.int32)

        # B x N x 3
        # Flipping the y-axis here to make it align with the image coordinate system!
        v[:, :, 1] *= -1

        if textures is not None:

            return self.renderer.render_rgb(v, f, textures)

        else:

            return self.renderer.render_silhouettes(v, f)
