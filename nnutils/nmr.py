from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import tqdm

import chainer
import torch

import neural_renderer as nr

from ..nnutils import geom_utils

#############
### Utils ###
#############
def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src

########################################################################
############ Wrapper class for the pytorch Neural Renderer #############
##### All functions must only use numpy arrays as inputs/outputs #######
########################################################################
class NMR(object):
    def __init__(self, cuda_device=None):
        # setup renderer
        self.renderer = renderer = nr.Renderer(
            camera_mode='look_at'
        )
        self.cuda_device = cuda_device

    def set_cuda(self, tensor):

        if self.cuda_device is not None:
            tensor = tensor.cuda(device=self.cuda_device)
        else:
            tensor = tensor.cpu()

        return tensor

    def make_tensor(self, numpy_array, dtype=None, requires_grad=False):

        dtype = (
            dtype
            if dtype is not None
            else getattr(torch, numpy_array.dtype.name)
        )

        tensor = self.set_cuda(
            torch.from_numpy(numpy_array).type(dtype)
        )

        if requires_grad:
            tensor = tensor.requires_grad_()

        return tensor

    def forward_mask(self, vertices, faces):
        # Renders masks.
        # Args:
        #     vertices: B X N X 3 numpy array
        #     faces: B X F X 3 numpy array
        # Returns:
        #     masks: B X 256 X 256 numpy array

        self.faces = self.make_tensor(faces, torch.IntTensor)
        self.vertices = self.make_tensor(vertices, requires_grad=True)
        #self.faces = chainer.Variable(chainer.cuda.to_gpu(faces, self.cuda_device))
        #self.vertices = chainer.Variable(chainer.cuda.to_gpu(vertices, self.cuda_device))

        import ipdb;ipdb.set_trace()
        self.masks = self.renderer.render_silhouettes(self.vertices, self.faces)

        return self.masks.cpu().numpy() # (16, 256, 256)

        masks = self.masks.data.get()
        return masks # numpy array (16, 256, 256)

    def backward_mask(self, grad_masks):
        # Compute gradient of vertices given mask gradients.
        # Args:
        #     grad_masks: B X 256 X 256 numpy array
        # Returns:
        #     grad_vertices: B X N X 3 numpy array

        import ipdb;ipdb.set_trace()
        self.masks.backward(gradient=self.make_tensor(grad_masks))

        return self.vertices.grad.numpy()

        return self.vertices.grad.get() # numpy array (16, 642, 3)

    def forward_img(self, vertices, faces, textures):
        # Renders masks.
        # Args:
        #     vertices: B X N X 3 numpy array
        #     faces: B X F X 3 numpy array
        #     textures: B X F X T X T X T X 3 numpy array
        # Returns:
        #     images: B X 3 x 256 X 256 numpy array

        self.faces = self.make_tensor(faces, torch.IntTensor)
        self.vertices = self.make_tensor(vertices, requires_grad=True)
        self.textures = self.make_tensor(textures, requires_grad=True)
        #self.faces = chainer.Variable(chainer.cuda.to_gpu(faces, self.cuda_device))
        #self.vertices = chainer.Variable(chainer.cuda.to_gpu(vertices, self.cuda_device))
        #self.textures = chainer.Variable(chainer.cuda.to_gpu(textures, self.cuda_device))

        self.images = self.renderer.render_rgb(self.vertices, self.faces, self.textures)

        import ipdb;ipdb.set_trace()
        return self.images.cpu().numpy() # numpy array (16, 3, 256, 256)

        images = self.images.data.get()
        return images # numpy array (16, 3, 256, 256)

    def backward_img(self, grad_images):
        # Compute gradient of vertices given image gradients.
        # Args:
        #     grad_images: B X 3? X 256 X 256 numpy array
        # Returns:
        #     grad_vertices: B X N X 3 numpy array
        #     grad_textures: B X F X T X T X T X 3 numpy array

        import ipdb;ipdb.set_trace()
        self.images.backward(gradient=self.make_tensor(grad_images))

        return self.vertices.grad.numpy(), self.textures.grad.numpy()

        return self.vertices.grad.get(), self.textures.grad.get() # numpy array (16, 642, 3), numpy array (16, 1280, 6, 6, 6, 3)

########################################################################
################# Wrapper class a rendering PythonOp ###################
##### All functions must only use torch Tensors as inputs/outputs ######
########################################################################
class Render(torch.autograd.Function):

    @staticmethod
    def forward(ctx, renderer, vertices, faces, textures=None):

        import ipdb;ipdb.set_trace()

        ctx.renderer = renderer

        # B x N x 3
        # Flipping the y-axis here to make it align with the image coordinate system!
        #vs = vertices.cpu().numpy()
        #vs[:, :, 1] *= -1
        #fs = faces.cpu().numpy()

        vertices[:, :, 1] *= -1
        ctx.mark_dirty(vertices)
        ctx.vertices = vertices

        if textures is None:

            ctx.masks = renderer.render_silhouettes(vertices, faces)
            return ctx.masks

        else:

            ctx.textures = textures
            ctx.mark_dirty(textures)

            ctx.images = renderer.render_rgb(vertices, faces, textures)
            return ctx.images

    @staticmethod
    def backward(ctx, grad_out):
        import ipdb;ipdb.set_trace()

        renderer = ctx.renderer
        vertices = ctx.vertices

        if hasattr(ctx, 'masks'):

            masks = ctx.masks

            masks.backward(gradient=grad_out)

            grad_verts = vertices.grad
            grad_tex = None

            del masks, ctx.masks

        else:

            textures = ctx.textures
            images = ctx.images

            images.backward(gradient=grad_out)

            grad_verts = vertices.grad
            grad_tex = textures.grad

            del textures, ctx.textures
            del images, ctx.images

        del renderer, ctx.renderer
        del vertices, ctx.vertices

        grad_verts[:, :, 1] *= -1
        return None, grad_verts, None, grad_tex

########################################################################
############## Wrapper torch module for Neural Renderer ################
########################################################################
class NeuralRenderer(torch.nn.Module):
    """
    This is the core pytorch function to call.
    Every torch NMR has a chainer NMR.
    Only fwd/bwd once per iteration.
    """
    def __init__(self, img_size=256, cuda_device=None):
        super(NeuralRenderer, self).__init__()
        self.renderer = nr.Renderer(
            camera_mode='look_at'
        )
        self.cuda_device = cuda_device

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

        if self.cuda_device is not None:
            new_tensor = new_tensor.cuda(device=self.cuda_device)
        else:
            new_tensor = new_tensor.cpu()

        return new_tensor

    def forward(self, vertices, faces, cams, textures=None):
        verts = self.proj_fn(vertices, cams, offset_z=self.offset_z)

        v = self._new_tensor(verts)
        f = faces.to(device=v.device, dtype=torch.int32)

        if textures is not None:
            t = self._new_tensor(textures)
            return Render.apply(self.renderer, v, f, t)
        else:
            return Render.apply(self.renderer, v, f)


########################################################################
############################## Tests ###################################
########################################################################
def exec_main():
    obj_file = 'birds3d/external/neural_renderer/examples/data/teapot.obj'
    vertices, faces = neural_renderer.load_obj(obj_file)

    renderer = NMR(cuda_device=0)

    masks = renderer.forward_mask(vertices[None, :, :], faces[None, :, :])
    print(np.sum(masks))
    print(masks.shape)

    grad_masks = masks*0 + 1
    vert_grad = renderer.backward_mask(grad_masks)
    print(np.sum(vert_grad))
    print(vert_grad.shape)

    # Torch API
    mask_renderer = NeuralRenderer()
    vertices_var = torch.from_numpy(vertices[None, :, :]).cuda(device=0).requires_grad_()
    faces_var = torch.from_numpy(faces[None, :, :]).cuda(device=0)
    
    for ix in range(100):
        masks_torch = mask_renderer.forward(vertices_var, faces_var)
        vertices_var.grad = None
        masks_torch.backward(torch.from_numpy(grad_masks).cuda(device=0))

    print(torch.sum(masks_torch))
    print(masks_torch.shape)
    print(torch.sum(vertices_var.grad))

def teapot_deform_test():
    obj_file = 'birds3d/external/neural_renderer/examples/data/teapot.obj'
    img_file = 'birds3d/external/neural_renderer/examples/data/example2_ref.png'
    img_save_dir = 'birds3d/cachedir/nmr/'

    vertices, faces = neural_renderer.load_obj(obj_file)

    image_ref = scipy.misc.imread(img_file).astype('float32').mean(-1) / 255.
    image_ref = torch.Tensor(image_ref[None, :, :]).cuda(device=0)

    mask_renderer = NeuralRenderer()
    faces_var = torch.from_numpy(faces[None, :, :]).cuda(device=0)
    cams = np.array([1., 0, 0, 1, 0, 0, 0], dtype=np.float32)
    cams_var = torch.from_numpy(cams[None, :]).cuda(device=0)

    class TeapotModel(torch.nn.Module):
        def __init__(self):
            super(TeapotModel, self).__init__()
            vertices_var = torch.from_numpy(vertices[None, :, :]).cuda(device=0)
            self.vertices_var = torch.nn.Parameter(vertices_var)

        def forward(self):
            return mask_renderer.forward(self.vertices_var, faces_var, cams_var)

    opt_model = TeapotModel()
    optimizer = torch.optim.Adam(opt_model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    # from time import time
    loop = tqdm.tqdm(range(300))
    print('Optimizing Vertices: ')
    for ix in loop:
        # t0 = time()
        optimizer.zero_grad()
        masks_pred = opt_model.forward()
        loss = torch.nn.MSELoss()(masks_pred, image_ref)
        loss.backward()
        if ix % 20 == 0:
            im_rendered = masks_pred.data.cpu().numpy()[0, :, :]
            scipy.misc.imsave(img_save_dir + 'iter_{}.png'.format(ix), im_rendered)
        optimizer.step()
        # t1 = time()
        # print('one step %g sec' % (t1-t0))

if __name__ == '__main__':
    # exec_main()
    teapot_deform_test()
