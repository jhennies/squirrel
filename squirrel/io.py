
import os
from h5py import File
from tifffile import imwrite
import numpy as np


def make_directory(directory, exist_ok=False, not_found_ok=False):
    try:
        os.mkdir(directory)
    except FileNotFoundError:
        if not_found_ok:
            return 'not_found'
        raise
    except FileExistsError:
        if exist_ok:
            return 'exists'
        raise
    return None


def load_h5_container(filepath, key, axes_order='zyx'):

    with File(filepath, mode='r') as f:
        data = f[key][:]

    if axes_order == 'zyx':
        return data

    axes_order_list = list(axes_order)
    return np.transpose(data, axes=[
        axes_order_list.index('z'),
        axes_order_list.index('y'),
        axes_order_list.index('x')
    ])


def write_tif_stack(data, out_folder):

    for idx, section in enumerate(data):
        im_filepath = os.path.join(out_folder, 'slice_{:04d}.tif'.format(idx))
        imwrite(im_filepath, section)
