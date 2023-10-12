
import os
from h5py import File
from tifffile import imwrite, imread
import numpy as np
from glob import glob


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
        write_tif_slice(section, out_folder, 'slice_{:04d}.tif'.format(idx))


def read_tif_slice(filepath):

    image = imread(filepath)

    # Return the image and the filename
    return image, os.path.split(filepath)[1]


def write_tif_slice(image, out_folder, filename):
    im_filepath = os.path.join(out_folder, filename)
    imwrite(im_filepath, image, compression='zlib')


def get_file_list(path, pattern='*'):

    return sorted(glob(os.path.join(path, pattern)))

