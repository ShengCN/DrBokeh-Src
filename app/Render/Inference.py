import numpy as np

import torch
from torchvision import utils
import matplotlib.pyplot as plt
import yaml
import cv2

import os
from os.path import join
import logging

import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, '.')
sys.path.insert(0, currentdir)

import argparse
import pandas as pd
from tqdm import tqdm
from image_preprocessing import RGBAD2layers, depth_predict, salient_segmentation
from app.Render.DScatter.scatter import Multi_Layer_Renderer
import traceback

class DrBokeh:
    def __init__(self, lens: int,  highlight=False, highlight_threshold=200.0/255.0, highlight_enhance_ratio=0.2):
        device = torch.device('cuda:0')
        self.render_model = Multi_Layer_Renderer(lens).to(device)

        self.highlight               = highlight
        self.highlight_threshold     = highlight_threshold
        self.highlight_enhance_ratio = highlight_enhance_ratio


    def inference(self, rgb: np.array, disp: np.array, alpha: np.array, params: dict):
        """ Do the bokeh rendering inference

        :param rgb:  RGB np.array. H x W x 3
        :param disp: disp np.array. H x W x 1
        :param alpha: alpha np.array. H x W x 1
        :param params: Rendering related paramters, {'focal': , 'lens_effect': , 'Salient'}
        :returns: bokeh image in np.array

        """
        assert rgb.shape[-1] == 3, 'Input rgb should have 3 channels({})'.format(rgb.shape[-1])
        assert disp.shape[-1] == 1, 'The last channel of the input disp should have 1 channels({})'.format(disp.shape[-1])
        assert alpha.shape[-1] == 1, 'The last channel of the input alpha should have 1 channels({})'.format(alpha.shape[-1])

        keys = ['focal', 'lens_effect', 'Salient']
        for k in keys:
            assert k in params, '{} should be in params'.format(k)

        focal                   = params['focal']
        lens_effect             = params['lens_effect']
        gamma                   = params['gamma']
        offset                  = params['offset']
        highlight               = self.highlight
        highlight_threshold     = self.highlight_threshold
        highlight_enhance_ratio = self.highlight_enhance_ratio

        layers = RGBAD2layers(rgb, alpha, disp, params)

        fg_rgbad = layers['fg_rgbad']
        bg_rgbad = layers['bg_rgbad']


        # highlight and gamma
        if highlight:
            mask1  = np.clip(np.tanh(200 * (np.abs(disp - focal)**2 - 0.01)), 0, 1)  # out-of-focus areas
            mask2  = np.clip(np.tanh(10*(rgb - highlight_threshold)), 0, 1)    # highlight areas
            mask   = mask1 * mask2

            fg_rgbad[..., :3] = fg_rgbad[..., :3] * (1 + mask * highlight_enhance_ratio)
            bg_rgbad[..., :3] = fg_rgbad[..., :3] * (1 + mask * highlight_enhance_ratio)

        fg_rgbad[..., :3] = (fg_rgbad[..., :3] + offset) ** gamma
        bg_rgbad[..., :3] = (bg_rgbad[..., :3] + offset) ** gamma

        input_x = {
            # 'layers': np.concatenate([fg_rgbad, bg_rgbad], axis=2),
            'layers': [fg_rgbad, bg_rgbad],
            'focal': focal,
            'lens_effect': lens_effect
        }

        # ret = self.model.inference(input_x)
        with torch.no_grad():
            bokeh = self.render_model.inference(input_x['layers'], lens_effect, focal)

        bokeh = bokeh ** (1.0/gamma) - offset

        ret = {
            'rgb': rgb,
            'disp': disp,
            'layers': layers,
            'bokeh': bokeh
        }

        return ret


    def single_layer_inference(self, rgb: np.array, disp: np.array, params: dict):
        """ Do the bokeh rendering inference

        :param rgb:  RGB np.array. H x W x 3
        :param disp: disp np.array. H x W x 1
        :param params: Rendering related paramters, {'focal': , 'lens_effect': , 'Salient'}
        :returns: bokeh image in np.array

        """
        assert rgb.shape[-1] == 3, 'Input rgb should have 3 channels({})'.format(rgb.shape[-1])
        assert disp.shape[-1] == 1, 'Input disp should have 1 channels({})'.format(disp.shape[-1])

        keys = ['focal', 'lens_effect', 'Salient']
        for k in keys:
            assert k in params, '{} should be in params'.format(k)

        focal                   = params['focal']
        lens_effect             = params['lens_effect']
        gamma                   = self.gamma
        highlight               = self.highlight
        highlight_threshold     = self.highlight_threshold
        highlight_enhance_ratio = self.highlight_enhance_ratio

        rgbad = np.concatenate([rgb, np.ones_like(disp), disp], axis=2)

        # highlight and gamma
        if highlight:
            mask1  = np.clip(np.tanh(200 * (np.abs(disp - focal)**2 - 0.01)), 0, 1)  # out-of-focus areas
            mask2  = np.clip(np.tanh(10*(rgb - highlight_threshold)), 0, 1)    # highlight areas
            mask   = mask1 * mask2
            rgbad[..., :3] = rgbad[..., :3] * (1 + mask * highlight_enhance_ratio)

        rgbad[..., :3] = rgbad[..., :3] ** gamma

        input_x = {
            # 'layers': np.concatenate([fg_rgbad, bg_rgbad], axis=2),
            'layers': [rgbad],
            'focal': focal,
            'lens_effect': lens_effect
        }

        # ret = self.model.inference(input_x)
        with torch.no_grad():
            bokeh = self.render_model.inference(input_x['layers'], lens_effect, focal)

        bokeh = bokeh ** (1.0/gamma)

        ret = {
            'rgb': rgb,
            'disp': disp,
            'bokeh': bokeh
        }

        return ret


    def resize(self,img, size):
        h, w = img.shape[:2]

        if h > w:
            newh = size
            neww = int(w / h * size)
        else:
            neww = size
            newh = int(h / w * size)

        resized_img = cv2.resize(img, (neww, newh))
        if len(img.shape) != len(resized_img.shape):
            resized_img = resized_img[..., None]

        return resized_img


"""
----------------------------------------------------------------
"""
def parse_configs(config):
    with open(config, 'r') as stream:
        try:
            configs=yaml.safe_load(stream)
            return configs
        except yaml.YAMLError as exc:
            logging.error(exc)
            return {}


def load_model(model, weight, device):
    cp = torch.load(weight)

    models = model.get_models()
    for k, m in models.items():
        m.load_state_dict(cp[k])
        m.to(device)

    model.set_models(models)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DScatter Inference')
    parser.add_argument('--rgb', type=str, help='RGB input', required=True)
    parser.add_argument('-K','--K_blur', type=float, help='Blur strength, a scaling factor that affects how large the bokeh shape would be', required=True)
    parser.add_argument('--focal', type=float, help='Focal plane', required=True)
    parser.add_argument('--ofile', type=str, help='Bokeh output path', required=True)
    parser.add_argument('--verbose', action='store_true', help='Verbose mode, will save the middle image representation for debug')

    parser.add_argument('--lens', type=int, help='Biggest lens kernel size, i.e. the largest neighborhood region', default=41)
    parser.add_argument('--disp', type=str, help='[Optional] Disparity input. Better disparity map makes results better. If not given, we use DPT to predict the depth.', default='')
    parser.add_argument('--alpha', type=str, help='[Optional] Alpha map input. Better alpha map makes results better. If not given, we use LDF to segment the salient object.', default='')
    parser.add_argument('--gamma', type=float, help='Naive gamma correction', default=2.2)
    args = parser.parse_args()

    # lens parameters
    lens        = args.lens
    lens_effect = args.K_blur
    focal       = args.focal
    gamma       = args.gamma

    drbokeh        = DrBokeh(lens)
    salient_method = 'LDF' 

    # inputs/output files
    rgb_file    = args.rgb
    disp_file   = args.disp
    alpha_file  = args.alpha
    output_path = args.ofile
    verbose     = args.verbose

    assert os.path.exists(rgb_file), 'RGB file does not exist'

    # RGB 
    rgb = plt.imread(rgb_file)[..., :3]

    # Disp
    if disp_file != '' and os.path.exists(disp_file):
        try:
            print('Read disp')
            disp = np.load(disp_file)['data']
            assert len(disp.shape) == 2, 'disp should have 2 channel'
        except Exception as e:
            print('Note, disp file is assumed to be a npz file with key \"data\"')
            traceback.print_exc()
    else:
        # predicted depth 
        print('Predicting disp')
        disp = depth_predict(rgb)

    disp = (disp - disp.min()) / (disp.max() - disp.min())

    # Alpha
    if alpha_file != '' and os.path.exists(alpha_file):
        try:
            print('Read alpha')
            alpha = plt.imread(alpha_file)[...,-1]
            assert len(alpha.shape) == 2, 'disp should have 2 channel'
        except Exception as e:
            print('Note, alpha file is assumed to be a npz file with key \"data\"')
            traceback.print_exc()
    else:
        # predicted alpha 
        print('Predicting alpha')
        alpha = salient_segmentation(rgb)

    # make sure the rgb, disp and alpha matches the size, align disp with RGB
    h, w = rgb.shape[:2]
    alpha = cv2.resize(alpha, (w, h))[..., None]
    disp = cv2.resize(disp, (w, h))[..., None]

    # Some hyper-params 
    params = {
        'focal': focal,
        'lens_effect': lens_effect,
        'Salient': salient_method,
        'gamma': gamma,
        'offset': 0.0,
        'fg_erode': 5,
        'fg_iters': 2,
        'inpaint_kernel': 7,
        'threshold': 0.1,     
        'Occlusion': True
    }

    # Render
    ret = drbokeh.inference(rgb, disp, alpha, params)
    layers   = ret['layers']
    fg_rgbad = layers['fg_rgbad']
    bg_rgbad = layers['bg_rgbad']

    # Save 
    odir = os.path.dirname(output_path)
    os.makedirs(odir, exist_ok=True)
    plt.imsave(output_path, np.ascontiguousarray(ret['bokeh']))

    print('Refer to ', output_path)

    if verbose:
        prefix = os.path.splitext(output_path)[0] 
        plt.imsave(f'{prefix}_DScatter_fg_rgba.png', np.ascontiguousarray(fg_rgbad[..., :4]))
        plt.imsave(f'{prefix}_DScatter_fg_disp.png', np.ascontiguousarray(fg_rgbad[..., -1]), cmap='plasma')
        plt.imsave(f'{prefix}_DScatter_bg_rgba.png', np.ascontiguousarray(bg_rgbad[..., :4]))
        plt.imsave(f'{prefix}_DScatter_bg_disp.png', np.ascontiguousarray(bg_rgbad[..., -1]), cmap='plasma')
