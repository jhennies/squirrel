
import os

import h5py
from h5py import File
from tifffile import imwrite, imread
import numpy as np
from glob import glob
from .data import invert_data


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


def load_h5_container(filepath, key, axes_order='zyx', invert=False):

    with File(filepath, mode='r') as f:
        data = f[key][:]

    if invert:
        data = invert_data(data)

    if axes_order == 'zyx':
        return data

    axes_order_list = list(axes_order)
    return np.transpose(data, axes=[
        axes_order_list.index('z'),
        axes_order_list.index('y'),
        axes_order_list.index('x')
    ])


def write_h5_container(filepath, data, key='data', append=False):
    # TODO add test

    mode = 'a' if append else 'w'

    with File(filepath, mode=mode) as f:
        f.create_dataset(key, data=data, compression='gzip')


def write_tif_stack(data, out_folder):

    for idx, section in enumerate(data):
        write_tif_slice(section, out_folder, 'slice_{:04d}.tif'.format(idx))


def read_tif_slice(filepath, return_filepath=True):

    image = imread(filepath)

    # Return the image and the filename
    if return_filepath:
        return image, os.path.split(filepath)[1]
    return image


def write_tif_slice(image, out_folder, filename):
    im_filepath = os.path.join(out_folder, filename)
    imwrite(im_filepath, image, compression='zlib')


def get_file_list(path, pattern='*'):
    return sorted(glob(os.path.join(path, pattern)))


def load_tif_stack(path, pattern='*'):
    return [read_tif_slice(filepath, return_filepath=False) for filepath in get_file_list(path, pattern)]


def write_nii_file(data, filepath, scale=None):
    # FIXME add tests
    import nibabel as nib
    affine = np.eye(4)
    if scale is not None:
        affine = np.eye(4)
        affine[0, 2] = scale[0]
        affine[1, 2] = scale[1]
        affine[2, 2] = scale[2]
    data = nib.Nifti1Image(data, affine=affine)
    nib.save(data, filepath)


def load_nii_file(filepath, invert=False):
    # FIXME add tests
    import nibabel as nib
    data = np.array(nib.load(filepath).dataobj)
    if invert:
        return invert_data(data)
    return data


def get_filetype(filepath):
    # FIXME add tests

    h5_extensions = ['.h5', '.H5', '.hdf5', '.HDF5']
    nii_extensions = ['.nii']
    tif_extensions = ['.tif', '.TIF', '.tiff', '.TIFF']
    json_extensions = ['.json', '.JSON']
    csv_extensions = ['.csv', '.CSV']

    ext = os.path.splitext(filepath)[1]
    if os.path.isdir(filepath):
        return 'dir'
    if ext in h5_extensions:
        return 'h5'
    if ext in nii_extensions:
        return 'nii'
    if ext in tif_extensions:
        return 'tif'
    if ext in json_extensions:
        return 'json'
    if ext in csv_extensions:
        return 'csv'
    raise ValueError(f'Unknown file extension: {ext}')


def load_data(filepath, key='data', axes_order='zyx', invert=False):

    filetype = get_filetype(filepath)

    if filetype == 'h5':
        return load_h5_container(filepath, key, axes_order=axes_order, invert=invert)
    if filetype == 'nii':
        return load_nii_file(filepath, invert=invert)
    if filetype == 'tif':
        raise NotImplementedError('Not implemented for 3D tif files')
    if filetype == 'dir':
        return np.array(load_tif_stack(filepath))
    raise RuntimeError(f'Invalid or unknown file type: {filetype}')


def load_data_from_handle_stack(h, idx):
    """
    :param h:
    :param idx:
    :return: np.array(data_slice), slice_filepath
    Note that slice_filepath is None if it's a hdf5 file handle
    """

    if type(h[idx]) == str:
        assert get_filetype(h[idx]) == 'tif'
        return read_tif_slice(h[idx])

    # Assuming h5 file handle
    return h[idx], None


def load_data_handle(path, key='data', pattern='*.tif'):

    filetype = get_filetype(path)

    if filetype == 'h5':
        h = h5py.File(path, mode='r')[key]
        return h, h.shape

    if filetype == 'dir':
        h = sorted(glob(os.path.join(path, pattern)))
        shape = load_data_from_handle_stack(h, 0)[0].shape
        return h, [len(h)] + list(shape)



