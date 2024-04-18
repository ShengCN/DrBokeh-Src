import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)

from salient import Salient_Inference
from rgb_inpainting import RGB_Inpainting_Inference
from depth_predict import Depth_Inference
import bilateral_median_filter

import cv2


def RGBAD2layers(rgb: np.array, mask: np.array, disp: np.array, params):
    """ Given RGBAD, i.e. RGBD + Mask as input

    :param rgb:   H x W x 3, np.array for RGB
    :param mask:  H x W x 1, np.array for mask
    :param disp:  H x W x 1, np.array for disp
    :returns:

    """
    assert rgb.shape[-1]  == 3, 'RGB should have 3 channels({})'.format(rgb.shape[-1])
    assert mask.shape[-1] == 1, 'mask should have 1 channels({})'.format(mask.shape[-1])
    assert disp.shape[-1] == 1, 'disp should have 1 channels({})'.format(disp.shape[-1])

    h, w = rgb.shape[:2]

    threshold      = params['threshold']
    fg_erode       = params['fg_erode']
    fg_iters       = params['fg_iters']
    inpaint_kernel = params['inpaint_kernel']

    fg_rgbad = process_fg(rgb, mask, disp, threshold=threshold, fg_erode=fg_erode, iterations=fg_iters, filter_input=False, inpaint_kernel=inpaint_kernel)
    bg_rgbad = process_bg(rgb, mask, disp, threshold=threshold, filter_input=False)

    return {
        'fg_rgbad': fg_rgbad,
        'bg_rgbad': bg_rgbad,
    }


def RGBD2layers(rgb: np.array, disp: np.array, params: dict):
    """ Given the rgb, depth image,
          1. Transform depth to disparity
          2. Do the sepration
          3. Background inpainting operation

    :param rgb:    H x W x C np.array for RGB
    :param disp:   H x W x 1 np.array for disp
    :param params: {'Salient'}
    :returns: {'fg_rgbad': , 'bg_rgbad': }

    """
    assert rgb.shape[-1] == 3, 'Input rgb should have 4 channels({})'.format(rgb.shape[-1])
    assert disp.shape[-1] == 1, 'Input rgb should have 4 channels({})'.format(disp.shape[-1])
    assert 'Salient' in params, 'Which Salient algorithm?'

    h, w         = rgb.shape[:2]
    rgbd         = np.concatenate([rgb, disp], axis=2)

    salient_mask = salient_segmentation(rgbd, params)
    if len(salient_mask.shape) == 2:
        salient_mask = salient_mask[..., None]

    return RGBAD2layers(rgb, salient_mask, disp, params)


def RGB2layers(rgb: np.array, params: dict):
    """ Given the RGB channels, we first predict the depth. Then pass the RGBD to later stage for processing.

    :param rgb: H x W x 3 np.array for RGB channels
    :param params: {'Salient'}
    :returns: RGB, disp

    """
    assert rgb.shape[-1] == 3, 'Input rgb should have 4 channels({})'.format(rgb.shape[-1])
    assert 'Salient' in params, 'Which Salient algorithm?'

    disp = depth_predict(rgb)

    if len(disp.shape) == 2:
        disp = disp[..., None]

    disp = (disp-disp.min())/(disp.max()-disp.min())

    return RGBD2layers(rgb, disp, params)


"""
----------------------------------------------------------------
"""
def preprocess_mask(mask: np.array, threshold=0.1, filter_input=False, kernel_size=11, sigma_s=4.0, sigma_r=0.9):
    """ Bilateral Median Filtering

    Ref: https://cs.brown.edu/courses/csci1290/labs/lab_bilateral/index.html
         https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=b086cfe529d132feb7accf10f4c35555bf9a96bb

    :param mask: H x W x 1 np.array
    :returns: filtered mask: H x W x 1 np.array

    """
    assert kernel_size % 2 == 1, 'Kernel should be odd number'

    if filter_input:
        ret = bilateral_median_filter(mask, kernel_size, sigma_s, sigma_r)
    else:
        ret = mask.copy()

    threshold          = threshold
    ret[ret<threshold] = 0.0
    ret[ret>threshold] = 1.0

    if len(ret.shape) == 2:
        ret = ret[..., None]

    return ret


def process_fg(rgb, mask, disp, threshold=0.1, fg_erode=5, iterations=3, filter_input=False, inpaint_kernel=11):
    """ Given the scene, we need to split the scene and compute the foreground layer

    Some times tips:
        1. smooth the boundary depth, so we shrink-in a little bit

    :param rgb:     H x W x 3 np.array
    :param mask:    H x W x 1 np.array
    :param disp:    H x W x 1 np.array
    :returns:       H x W x 5 np.array

    """
    h, w = rgb.shape[:2]

    hard_mask = preprocess_mask(mask, threshold, filter_input)

    # inpaint_mask = erode(hard_mask, size=fg_erode, iterations=iterations)
    inpaint_mask = dilate(1.0-hard_mask, size=fg_erode, iterations=iterations)

    if inpaint_mask.shape != disp.shape:
        inpaint_mask = cv2.resize(inpaint_mask, (w, h))

    if len(inpaint_mask.shape) == 2:
        inpaint_mask = inpaint_mask[..., None]

    # inpaint_disp = disp * inpaint_mask
    # inpaint_mask = ((1.0-inpaint_mask) * 255.0).astype(np.uint8)
    inpaint_mask = (inpaint_mask * 255.0).astype(np.uint8)
    inpaint_disp = cv2.inpaint((disp * 65535).astype(np.uint16),
                               inpaint_mask,
                               inpaint_kernel,
                               cv2.INPAINT_TELEA) / 65535.0
    inpaint_disp = inpaint_disp[..., None]

    # inpaint_disp = depth_inpainting(inpaint_disp, 1.0-inpaint_mask)

    fg_rgbad = np.concatenate([rgb, mask, inpaint_disp], axis=2)
    # fg_rgbad = np.concatenate([rgb, hard_mask, inpaint_disp], axis=2)

    return fg_rgbad


def process_bg(rgb, mask, disp, threshold, filter_input=False):
    """ Given the scene, we need to split the scene and compute the background layer

    Some times tips:
        1. Do inpainting for the background

    :param rgb:     H x W x 3 np.array
    :param mask:    H x W x 1 np.array
    :param disp:    H x W x 1 np.array
    :returns:       H x W x 5 np.array

    """
    # maybe we need to enlarge some region
    h, w = rgb.shape[:2]

    hard_mask           = preprocess_mask(mask, threshold, filter_input)
    binary_salient_mask = hard_mask.copy()
    # binary_salient_mask[salient_mask>0.001] = 1.0
    # binary_salient_mask[salient_mask<0.001] = 0.0
    enlarge_mask = dilate(binary_salient_mask, size=5)
    bg_mask      = 1.0-enlarge_mask
    bg_rgb       = rgb * bg_mask
    bg_depth     = disp * bg_mask

    bg_rgb   = RGB_inpainting(bg_rgb, enlarge_mask)
    bg_depth = depth_inpainting(bg_depth, enlarge_mask)

    # bg_depth = cv2.GaussianBlur(bg_depth, (5, 5),0)
    # if len(bg_depth.shape) == 2:
    #     bg_depth = bg_depth[..., None]

    bg_rgbad = np.concatenate([bg_rgb, np.ones((h, w, 1)), bg_depth], axis=2)
    return bg_rgbad


salient_inferencer   = Salient_Inference()
def salient_segmentation(rgb: np.array):
    """ Run the LDF salient algorithm

    :param rgb: H x W x 3 input image
    :returns: H x W x C salient map

    """
    return salient_inferencer.inference(rgb)



rgb_inpainting_inferencer = RGB_Inpainting_Inference()
def RGB_inpainting(rgb: np.array, mask: np.array):
    """ RGB inpainting by LaMa

    :param rgb:  H x W x 3 RGB image
    :param mask: H x W x 1 alpha mask
    :returns: The inpainted image

    """
    inpainted = rgb_inpainting_inferencer.inference(rgb, mask)
    return inpainted


depth_inferencer = Depth_Inference()
def depth_predict(rgb: np.array):
    return depth_inferencer.inference(rgb)


def depth_inpainting(depth, mask):
    """ Depth Inpainting by cv2 inpaint

    :param depth: H x W x 1 depth image
    :param mask: H x W x 1 inpainting mask image
    :returns: Inpainted depth image

    """
    assert depth.shape[-1] == 1 and mask.shape[-1] == 1, \
        'depth and mask channels should be 1({}) and 1({}).'.format(depth.shape[-1], mask.shape[-1])

    tmp_depth = np.repeat(depth, 3, axis=2)
    inpainted = rgb_inpainting_inferencer.inference(tmp_depth, mask)[..., :1]
    return inpainted


def dilate(img, size=5, iterations=2):
    assert img.shape[-1] == 1, 'Dilation assumes img has 3 channels({})'.format(img.shape[-1])

    kernel       = np.ones((size, size), np.float64)
    img_dilation = cv2.dilate(img, kernel, iterations=iterations)
    return img_dilation[..., None]


def erode(img, size=5, iterations=2):
    assert img.shape[-1] == 1, 'Dilation assumes img has 3 channels({})'.format(img.shape[-1])

    kernel = np.ones((size, size), np.float64)
    img_dilation = cv2.erode(img, kernel, iterations=iterations)
    return img_dilation[..., None]




if __name__ == '__main__':
    import os
    from os.path import join
    import sys
    import inspect
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    tmp_out    = join(currentdir, 'out')
    os.makedirs(tmp_out, exist_ok=True)

    test_imgs = ['00156','00032']

    for img_name in tqdm(test_imgs):
        test_rgb   = plt.imread('app/Paper/Blur_Compare/img/{}.jpg'.format(img_name))/255.0
        test_depth = np.load('app/Paper/Blur_Compare/img/{}.npz'.format(img_name))['data']
        rgbd       = np.concatenate([test_rgb, test_depth[..., None]], axis=2)
        separation = img_preprocess(rgbd)

        fg_rgbad = separation['fg_rgbad']
        bg_rgbad = separation['bg_rgbad']

        print('fg: {}/{}, bg: {}/{}'.format(fg_rgbad.min(), fg_rgbad.max(), bg_rgbad.min(), bg_rgbad.max()))
        plt.imsave(join(tmp_out, '{}_fg_rgb.png'.format(img_name)), np.ascontiguousarray(fg_rgbad[..., :-1]))
        plt.imsave(join(tmp_out, '{}_fg_depth.png'.format(img_name)), np.ascontiguousarray(fg_rgbad[..., -1]), cmap='plasma')
        plt.imsave(join(tmp_out, '{}_bg_rgb.png'.format(img_name)), np.ascontiguousarray(bg_rgbad[..., :-1]))
        plt.imsave(join(tmp_out, '{}_bg_depth.png'.format(img_name)), np.ascontiguousarray(bg_rgbad[..., -1]), cmap='plasma')
