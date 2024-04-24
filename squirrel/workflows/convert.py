import os

from squirrel.library.io import make_directory, load_h5_container, write_tif_stack
from squirrel.library.io import get_file_list, read_tif_slice, write_tif_slice

import numpy as np


def h5_to_tif_workflow(
        h5_file,
        h5_key,
        out_folder,
        axes_order='zyx',
        verbose=False
):
    if verbose:
        print(f'h5_file = {h5_file}')
        print(f'h5_key = {h5_key}')
        print(f'out_folder = {out_folder}')

    # Create target folder (parent folder has to exist)
    make_directory(out_folder)

    # Load data from the source file
    data = load_h5_container(h5_file, h5_key, axes_order=axes_order)

    # Write results
    write_tif_stack(data, out_folder)


def h5_to_nii_workflow(
        h5_file,
        h5_key,
        out_filepath,
        axes_order='zyx',
        verbose=False
):
    from squirrel.library.io import write_nii_file

    if verbose:
        print(f'h5_file = {h5_file}')
        print(f'h5_key = {h5_key}')
        print(f'out_filepath = {out_filepath}')

    # Load data from the source file
    data = load_h5_container(h5_file, h5_key, axes_order=axes_order)

    # Write results
    write_nii_file(data, out_filepath, scale=None)


def mib_to_tif_workflow(
        mib_model_file,
        out_folder,
        verbose=False
):
    if verbose:
        print(f'mib_model_file = {mib_model_file}')
        print(f'out_folder = {out_folder}')

    h5_to_tif_workflow(
        mib_model_file,
        'mibModel',
        out_folder,
        axes_order='zxy',
        verbose=verbose
    )


def compress_tif_stack_workflow(
        in_folder,
        out_folder,
        pattern='*.tif',
        verbose=False
):
    if verbose:
        print(f'in_folder = {in_folder}')
        print(f'out_folder = {out_folder}')
        print(f'pattern = {pattern}')

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    im_list = get_file_list(in_folder, pattern)

    for im_filepath in im_list:

        if verbose:
            print(f'im_filepath = {im_filepath}')

        image, filename = read_tif_slice(im_filepath)
        write_tif_slice(image, out_folder, filename)


def merge_tif_stacks_workflow(
        stack_folders,
        out_folder,
        pattern='*.tif',
        out_pattern='slice_{:05d}.tif',
        pad_canvas=False,
        verbose=False
):
    if verbose:
        print(f'stack_folders = {stack_folders}')
        print(f'out_folder = {out_folder}')

    from ..library.io import get_filetype
    assert get_filetype(stack_folders[0]) == get_filetype(stack_folders[1]) == 'dir'

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    if pad_canvas:

        from ..library.io import load_data_handle, write_tif_slice
        from ..library.image import image_to_shape
        shapes = []
        handles = []
        for stack in stack_folders:
            h, s = load_data_handle(stack, pattern=pattern)
            shapes.append(s[1:])  # only y and x
            handles.append(h)
        new_shape = np.max(shapes, axis=0)
        print(f'new_shape = {new_shape}')
        idx = 0
        for h in handles:
            for img in h[:]:
                print(f'idx = {idx}')
                write_tif_slice(
                    image_to_shape(img, new_shape),
                    out_folder,
                    out_pattern.format(idx)
                )
                idx += 1

        return

    im_list = []
    for idx, folder in enumerate(stack_folders):
        im_list.extend(
            get_file_list(folder, pattern=pattern if type(pattern) == str or len(pattern) == 1 else pattern[idx])
        )

    from shutil import copyfile

    for idx, im_filepath in enumerate(im_list):

        if verbose:
            print(f'im_filepath = {im_filepath}')

        out_filepath = os.path.join(out_folder, out_pattern.format(idx))
        copyfile(im_filepath, out_filepath)


def stack_to_ome_zarr_workflow(
        stack_path,
        ome_zarr_filepath,
        stack_pattern='*.tif',
        stack_key='data',
        resolution=(1., 1., 1.),
        unit='pixel',
        downsample_type='Average',
        downsample_factors=(2, 2, 2),
        name=None,
        chunk_size=(64, 64, 64),
        z_range=None,
        save_bounds=False,
        append=False,
        n_threads=1,
        verbose=False
):

    if verbose:
        print(f'stack_path = {stack_path}')
        print(f'ome_zarr_filepath = {ome_zarr_filepath}')
        print(f'stack_pattern = {stack_pattern}')
        print(f'stack_key = {stack_key}')
        print(f'chunk_size = {chunk_size}')
        print(f'z_range = {z_range}')
        print(f'n_threads = {n_threads}')

    from squirrel.library.data import norm_z_range

    # Load the stack slices
    from ..library.io import load_data_handle
    input_stack_handle, input_stack_shape = load_data_handle(stack_path, key=stack_key, pattern=stack_pattern)

    z_range = norm_z_range(z_range, input_stack_shape[0])
    chunk_data = input_stack_handle[z_range[0]: z_range[1]]

    # Create ome zarr if necessary
    if not append:
        from ..library.ome_zarr import create_ome_zarr
        create_ome_zarr(
            ome_zarr_filepath,
            shape=input_stack_shape,
            resolution=resolution,
            unit=unit,
            downsample_type=downsample_type,
            downsample_factors=downsample_factors,
            chunk_size=chunk_size,
            dtype=input_stack_handle[0].dtype,
            name=name
        )

    # Write the stack to ome_zarr
    from squirrel.library.ome_zarr import chunk_to_ome_zarr, get_ome_zarr_handle
    chunk_to_ome_zarr(
        chunk_data,
        [z_range[0], 0, 0],
        get_ome_zarr_handle(ome_zarr_filepath, mode='a'),
        key='s0',
        populate_downsample_layers=True,
        verbose=verbose
    )


def ome_zarr_to_stack_workflow(
        ome_zarr_filepath,
        target_dirpath,
        ome_zarr_key='s0',
        z_range=None,
        n_threads=1,
        verbose=False
):

    if verbose:
        print(f'ome_zarr_filepath = {ome_zarr_filepath}')
        print(f'target_dirpath = {target_dirpath}')
        print(f'ome_zarr_key = {ome_zarr_key}')
        print(f'z_range = {z_range}')
        print(f'n_threads = {n_threads}')

    # Load the ome zarr
    from squirrel.library.data import norm_z_range
    from squirrel.library.ome_zarr import get_ome_zarr_handle
    handle = get_ome_zarr_handle(ome_zarr_filepath, ome_zarr_key, mode='r')
    z_range = norm_z_range(z_range, handle.shape[0])
    chunk_data = handle[z_range[0]: z_range[1], :]

    # Save to the tif stack
    if not os.path.exists(target_dirpath):
        os.mkdir(target_dirpath)
    from squirrel.library.io import write_tif_stack
    write_tif_stack(chunk_data, target_dirpath, id_offset=z_range[0], slice_name='slice_{:05d}.tif')

