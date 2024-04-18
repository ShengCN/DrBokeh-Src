import time
import logging
import matplotlib.pyplot as plt
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils, transforms
import numpy as np

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)

import CPU_scatter
import GPU_scatter
from CPU_scatter import distance_kernel, lens_shape_mask


class Multi_Layer_Renderer(nn.Module):
    def __init__(self, lens_mask, use_cuda=True, gpu_occlusion=True):
        super().__init__()

        self.gpu_occlusion = gpu_occlusion
        self.renderer = Scatter_Rendering(lens_mask, use_cuda, gpu_occlusion)


    def forward(self, rgbad_layers, lens_effect, focal):
        """
        Render the lens blur given the multi-layer representation.
        Note, the disparity layer is the relative disparity!

        @param rgbad_layers:  [B, 5 * n, H, W] float tensor (n: layer #)
        @param lens_effects: [B, 1] float tensor
        @param focal: [B, 5 * n, H, W] float tensor

        @return: blur: [B, 3, H, W]
        """
        n_layer = rgbad_layers.shape[1] // 5

        assert rgbad_layers.shape[1] % 5 == 0, "Layer number needs to be multiple of 5({})".format(rgbad_layers.shape[1] % 5)
        assert n_layer >= 0, "Layer number needs to be greater than 0"
        assert focal.shape == rgbad_layers.shape, 'rgbad_layers should have the same shape with focal'
        assert (rgbad_layers.dtype == torch.float) and (lens_effect.dtype == torch.float), \
                "Blur rendering assumes input tensors to be float"

        eps = 1e-6
        rgbad_layers = rgbad_layers - focal

        return self.defocus_render(rgbad_layers, lens_effect)


    def defocus_render(self, rgbad_layers, lens_effect):
        """

        Render the lens blur given the multi-layer representation.
        Note, the disparity layer is the relative disparity!

        @param rgbad_layers:  [B, 5 * n, H, W] float tensor (n: layer #)
        @param lens_effects: [B, 1] float tensor
        @param focal: [B, 5 * n, H, W] float tensor

        @return: blur: [B, 3, H, W]
        """
        if not self.gpu_occlusion:
            return self.no_occlusion_defocus_render(rgbad_layers, lens_effect)

        n_layer = rgbad_layers.shape[1] // 5

        assert rgbad_layers.shape[1] % 5 == 0, "Layer number needs to be multiple of 5({})".format(rgbad_layers.shape[1] % 5)
        assert n_layer >= 0, "Layer number needs to be greater than 0"
        assert (rgbad_layers.dtype == torch.float) and (lens_effect.dtype == torch.float), \
                "Blur rendering assumes input tensors to be float"

        eps = 1e-8
        blur_list = [self.renderer.forward(rgbad_layers[:, 5 * i:5 * (i+1)], lens_effect)
                     for i in range(n_layer)]

        # we need to render the blur from the first layer to the last layer
        blur_rgb  = blur_list[0][:, :-2]
        blur_w    = blur_list[0][:, -2:-1]
        blur_occu = blur_list[0][:, -1:]

        blur_rgb = blur_rgb / (blur_w + eps) * blur_occu
        blur_occu = 1.0-blur_occu

        for li in range(1, n_layer):
            layer_rgb  = blur_list[li][:, :-2]
            layer_w    = blur_list[li][:, -2:-1]
            layer_occu = blur_occu * blur_list[li][:, -1:]

            layer_blur = layer_rgb / (layer_w + eps) * layer_occu
            blur_occu = blur_occu * (1.0 - blur_list[li][:, -1:])

            blur_rgb = blur_rgb + layer_blur

        return blur_rgb


    def inference(self, rgbad_list: [np.array], lens_effect: float, focal: float):
        """
        Given a list of RGBAD layer, lens_effect and focal, render the lens blur effects.

        @param rgbad_list: A list of RGBAD
        @param lens_effect: Lens blur strength
        @param focal: Focal length

        @return: A numpy array, lens blur result
        """
        assert len(rgbad_list) > 0, "The RGBAD list input is empty"
        for i in range(len(rgbad_list)):
            assert rgbad_list[i].shape[2] == 5, "The RGBAD list elements should have 5 channel"

        rgbad_layer = torch.cat([torch.tensor(rgbad.transpose(2,0,1))[None, ...] for rgbad in rgbad_list], dim=1).cuda().float()
        lens_effect_tensor = (torch.ones(1, 1) * lens_effect).cuda().float()

        focal_tensor = torch.zeros_like(rgbad_layer).to(rgbad_layer)
        b,c,_,_ = rgbad_layer.shape
        for i in range(c//5):
            focal_tensor[:, i * 5 - 1] = focal


        blur = self.forward(rgbad_layer, lens_effect_tensor, focal_tensor)
        blur = blur[0].detach().cpu().numpy().transpose((1, 2, 0))

        return blur


    def no_occlusion_defocus_render(self, rgbad_layers, lens_effect):
        """

        Render the lens blur given the multi-layer representation.
        Note, the disparity layer is the relative disparity!

        @param rgbad_layers:  [B, 5 * n, H, W] float tensor (n: layer #)
        @param lens_effects: [B, 1] float tensor
        @param focal: [B, 5 * n, H, W] float tensor

        @return: blur: [B, 3, H, W]
        """
        n_layer = rgbad_layers.shape[1] // 5

        assert rgbad_layers.shape[1] == 5, "Layer number needs to be 5({})".format(rgbad_layers.shape[1])
        assert n_layer >= 0, "Layer number needs to be greater than 0"
        assert (rgbad_layers.dtype == torch.float) and (lens_effect.dtype == torch.float), \
                "Blur rendering assumes input tensors to be float"

        rgbd_layer = torch.cat([rgbad_layers[:,:3], 
                                rgbad_layers[:,-1:]], dim=1) 
        eps = 1e-8
        bokeh = self.renderer.forward(rgbd_layer, lens_effect)
        return bokeh


class Scatter_Rendering(nn.Module):
    """ Scatter Rendering Layer
            depth + scattering blur
    """

    def __init__(self, lens_mask, use_cuda=True, gpu_occlusion=True):
        if lens_mask % 2 == 0:
            raise ValueError("Lens mask {} is even".format(lens_mask))
        super(Scatter_Rendering, self).__init__()

        self.lens, self.padding = lens_mask, torch.nn.ReplicationPad2d(lens_mask // 2)
        self.diskernel = nn.Parameter(distance_kernel(self.lens).float(), requires_grad=False)
        self.lens_mask = nn.Parameter(lens_shape_mask(self.lens).float(), requires_grad=False)

        self.use_cuda = use_cuda
        # CUDA scatter function
        if self.use_cuda:
            if gpu_occlusion:
                self.scatter = GPU_scatter.Scatter.apply
            else:
                self.scatter = GPU_scatter.Scatter_no_occlusion.apply
        else:
            self.scatter = CPU_scatter.Scatter(self.lens)


    def forward(self, x, lens_effects):
        """ Scatteirng Rendering Layer
                x ~ B x 5 x H x W: 	relative disparity w.r.t. focal plane
        """
        b, c, h, w = x.shape
        assert c == 4 or c == 5, "Scattering Input is wrong. {}".format(c)

        if self.use_cuda:
            ret = self.scatter(x, lens_effects, self.diskernel, self.lens_mask)
        else:
            ret = self.scatter(x, lens_effects)

        # ret = ret/(ret[:, -1:] + 1e-8)
        return ret


    def inference(self, rgbd: np.array, lens_effect: float, focal: float):
        """
        Given the numpy array of RGBD, float lens_effect, render the results
        @param rgbd: RGBD numpy array. NOTE, the D is disparity
        @param lens_effect: lens strength
        @param focal: camera's focal length
        @return: blur image
        """
        h, w, c = rgbd.shape
        assert c == 4, 'Input channel should be 4({})'.format(c)

        rgb  = rgbd[..., :3]
        disp = rgbd[..., -1:] - focal
        rgbd = np.concatenate([rgb, disp], axis=2)

        rgbd_layer = torch.tensor(rgbd.transpose(2,0,1))[None, ...].cuda().float()
        lens_effect_tensor = (torch.ones(1, 1) * lens_effect).cuda().float()

        blur = self.forward(rgbd_layer, lens_effect_tensor)
        blur = blur[0].detach().cpu().numpy().transpose((1, 2, 0))

        return blur


if __name__ == '__main__':
    renderer = Multi_Layer_Renderer(21)
    h, w = 256, 256

    rgbad_layers = torch.randn(5, 5, 5, h, w).cuda()
    lens_effect  = torch.ones(5, 1).cuda()

    renderer = renderer.cuda()
    blur = renderer.forward(rgbad_layers, lens_effect, 0.3)
