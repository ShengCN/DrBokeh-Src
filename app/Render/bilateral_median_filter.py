import numpy as np
from functools import reduce
from tqdm import tqdm

def filter(img, kernel_size=11, sigma_s=4.0, sigma_r=0.5):
    img = np.squeeze(img)

    h, w        = img.shape[:2]
    pad_size    = kernel_size // 2
    padded_mask = np.pad(img, pad_size, 'edge')
    midpt       = pad_size

    ax           = np.arange(-midpt, midpt+1.)
    xx, yy       = np.meshgrid(ax, ax)
    spatial_term = np.exp(-(xx**2 + yy**2) / (2. * sigma_s**2))

    padded_maskh_patches = rolling_window(padded_mask, [kernel_size, kernel_size], [1,1])
    pH, pW = padded_maskh_patches.shape[:2]

    output = img.copy()
    for pi in tqdm(range(pH)):
        for pj in range(pW):
            patch       = padded_maskh_patches[pi, pj]
            depth_order = patch.ravel().argsort()
            patch_midpt = patch[kernel_size//2, kernel_size//2]
            range_term  = np.exp(-(patch-patch_midpt)**2 / (2. * sigma_r**2))

            coef = spatial_term * range_term

            if coef.sum() == 0:
                output[pi, pj] = patch_midpt
            else:
                coef = coef/(coef.sum())
                coef_order = coef.ravel()[depth_order]
                cum_coef = np.cumsum(coef_order)
                ind = np.digitize(0.5, cum_coef)
                output[pi, pj] = patch.ravel()[depth_order][ind]

    return output


"""
---------------------------------------------------------
"""
def rolling_window(a, window, strides):
    assert len(a.shape)==len(window)==len(strides), "\'a\', \'window\', \'strides\' dimension mismatch"
    shape_fn = lambda i,w,s: (a.shape[i]-w)//s + 1
    shape = [shape_fn(i,w,s) for i,(w,s) in enumerate(zip(window, strides))] + list(window)
    def acc_shape(i):
        if i+1>=len(a.shape):
            return 1
        else:
            return reduce(lambda x,y:x*y, a.shape[i+1:])
    _strides = [acc_shape(i)*s*a.itemsize for i,s in enumerate(strides)] + list(a.strides)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=_strides)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test   = plt.imread('img/test.png')[..., 0]

    kernel_size = 15
    sigma_r_ = [0.1, 0.5, 0.9, 1.0]
    sigma_s_ = [4, 8, 16, 32, 64]

    for sigma_r in sigma_r_:
        for sigma_s in sigma_s_:
            filted = filter(test, kernel_size=kernel_size, sigma_r = sigma_r, sigma_s = sigma_s)
            diff = np.abs(test-filted)
            print(diff.sum(), diff.min(), diff.max())

            plt.imsave('img/{:02d}_{:.3f}_{:.3f}_filted.png'.format(kernel_size, sigma_r, sigma_s), filted, cmap='gray')
            plt.imsave('img/{:02d}_{:.3f}_{:.3f}_diff.png'.format(kernel_size, sigma_r, sigma_s), diff)
