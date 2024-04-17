
import numpy as np


def invert_data(data):
    # FIXME add tests
    try:
        data = np.iinfo(data.dtype).max - data
    except ValueError:
        data = data.max() - data
    return data


def resolution_to_pixels(value, resolution):
    if type(value) in [list, tuple, np.array]:
        return np.array(value) / resolution
    return value / resolution


def norm_8bit(im, quantiles, ignore_zeros=False):
    im = im.astype('float32')
    if ignore_zeros:
        upper = np.quantile(im[im > 0], quantiles[1])
        lower = np.quantile(im[im > 0], quantiles[0])
    else:
        upper = np.quantile(im, quantiles[1])
        lower = np.quantile(im, quantiles[0])
    im -= lower
    im /= (upper - lower)
    im *= 255
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')


def norm_z_range(z_range, len_stack):

    if z_range is None:
        z_range = [0, len_stack]
    if z_range[1] > len_stack:
        z_range[1] = len_stack

    return z_range
