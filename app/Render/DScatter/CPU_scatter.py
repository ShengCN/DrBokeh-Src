""" Note, more details can be found in ${ProjectDir}/SDoF/Models/ScatterNet
"""
import os
from os.path import join
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import utils, transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging

def soft_step(a, b):
    """ Approxiate Differentiable a > b 
        Note, for stability, map a and b to 0.01 scale 
    """
    # span_range, grad_fact, domain_offset = 0.1, 3.0, 0.0
    # return span_range/(span_range+torch.exp(-(a-b+domain_offset)*grad_fact))
    s = 0.1
    g = 3.0 
    d = 0.0

    return s/(s + torch.exp(-(a-b+d)*g));

def distance_kernel(size):
    xgrid = (torch.arange(size)-size//2).repeat(size).view(size, size)
    disfield = torch.stack((xgrid, xgrid.T),dim=-1)
    return torch.sqrt(torch.pow(disfield[...,0],2) + torch.pow(disfield[...,1], 2))


def lens_shape_mask(size, shape='circle'):
    def circle(size):
        kernel = torch.zeros((size, size))
        diskernel = distance_kernel(size)
        kernel[diskernel<size//2] = 1.0
        return kernel
    lens_shape = {
        'circle':circle
    }
    if shape not in lens_shape.keys():
        raise ValueError("Shape input {} has not been implemented yet".format(shape))

    return lens_shape[shape](size)


def soft_occlusion(center_disp, neighbour_disp, lens_effects, is_center):
    if is_center:
        return torch.ones_like(neighbour_disp)

    rel_dis = neighbour_disp - center_disp
    center_scatter = torch.abs(center_disp) * lens_effects

    in_focal = 1.0-torch.exp(-center_scatter * center_scatter * 3.0)
    rel_occ = 0.5 + 0.5 * torch.tanh(10.0 * (rel_dis-0.1))

    occlusion = (1.0-in_focal) * rel_occ + in_focal

    return occlusion



class Scatter(nn.Module):
    """ Scatter Rendering Layer
            depth + scattering blur
    """
    def __init__(self, lens=21):
        if lens % 2 == 0:
            raise ValueError("Scattering lens is even")

        super(Scatter, self).__init__()
        self.lens, self.padding = lens, torch.nn.ReplicationPad2d(lens//2)
        self.diskernel, self.lens_mask = nn.Parameter(distance_kernel(self.lens)), nn.Parameter(lens_shape_mask(self.lens))

        # logging.info('Distance kernel: {}'.format(self.diskernel))
        # logging.info('Lens mask: {}'.format(self.lens_mask))

    def scatter_weights(self, lens_mask, lens_effect, disp, rel_dis, cur_disp, is_center):
        scatter_dis = torch.abs(disp) * lens_effect + 1.0

        # energy reweighting by disparities 
        area_reweights = 1./(scatter_dis * scatter_dis)

        weights = lens_mask * soft_step(scatter_dis, rel_dis) * area_reweights
        return  weights


    def soft_occlusion(self, center_disp, cur_disp, lens_effect, is_center):
        return soft_occlusion(center_disp, cur_disp, lens_effect, is_center)


    def forward(self, x, lens_effect):
        """ Scatteirng Rendering Layer 
                x ~ B x 5 x H x W:      relative disparity w.r.t. focal plane
        """
        b, c, h, w = x.shape
        if c != 5:
            raise ValueError("Scattering Input is wrong. {}".format(c))
        
        ret = torch.zeros((b, 5, h, w), requires_grad=True).to(x)

        # replicate padding
        paddedx, offset = self.padding(x), self.lens//2

        # for differentiability
        global_rgb_mask = torch.zeros_like(ret, requires_grad=False)
        for bi in range(b):
            for hi in tqdm(range(offset, h+offset)):
                for wi in range(offset, w+offset):
                    global_rgb_mask[...] = 0.0
                    center_disparity     = paddedx[bi, 4:, slice(hi, hi+1), slice(wi, wi+1)]

                    for ni in range(-offset, offset+1):
                        for nii in range(-offset, offset+1):
                            # slice neighbours indices
                            cur_global_slice = slice(hi+ni, hi+ni+1), slice(wi+nii,wi+nii+1)
                            kernel_slice_i = slice(ni+offset, ni+offset+1)
                            kernel_slice_j = slice(nii+offset, nii+offset+1)

                            # slice neighbour rgb and disparities
                            rgb_neighbours = paddedx[bi, :3, cur_global_slice[0], cur_global_slice[1]]
                            alpha          = paddedx[bi, 3:4, cur_global_slice[0], cur_global_slice[1]]
                            disparity      = paddedx[bi, 4:, cur_global_slice[0], cur_global_slice[1]]

                            # check scattering effects
                            lens = self.lens_mask[kernel_slice_i, kernel_slice_j]
                            distance = self.diskernel[kernel_slice_i, kernel_slice_j]

                            if ni == 0 and nii == 0:
                                weights = self.scatter_weights(lens, lens_effect, disparity, distance, center_disparity, True)
                                occlusion = self.soft_occlusion(center_disparity, disparity, lens_effect, True)
                            else:
                                weights = self.scatter_weights(lens, lens_effect, disparity, distance, center_disparity, False)
                                occlusion = self.soft_occlusion(center_disparity, disparity, lens_effect, False)

                            global_rgb_mask[bi,:3,hi-offset,wi-offset] += (rgb_neighbours * weights * occlusion * alpha).sum(dim=1).sum(dim=1)
                            global_rgb_mask[bi,3:4,hi-offset,wi-offset] += (weights * occlusion * alpha).sum(dim=1).sum(dim=1)

                    scatter_dis     = torch.abs(center_disparity) * lens_effect + 1.0
                    half_dis        = (scatter_dis // 2).int()
                    half_dis = min(offset, half_dis)

                    for ni in range(-half_dis, half_dis + 1):
                        for nii in range(-half_dis, half_dis + 1):
                            cur_global_slice = slice(hi+ni, hi+ni+1), slice(wi+nii,wi+nii+1)
                            alpha = paddedx[bi, 3:4, cur_global_slice[0], cur_global_slice[1]]
                            global_rgb_mask[bi,4:,hi-offset:hi-offset+1,wi-offset:wi-offset+1] += alpha

                    global_rgb_mask[bi,4:,hi-offset,wi-offset] = global_rgb_mask[bi,4:,hi-offset,wi-offset]/((half_dis * 2 + 1) * (half_dis * 2 + 1))

                    ret = ret + global_rgb_mask

        return ret
