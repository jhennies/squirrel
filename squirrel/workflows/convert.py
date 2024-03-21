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

    if pad_canvas:
        raise NotImplementedError('Canvas padding not implemented!')

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

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
        chunk_size=(1, 256, 256),
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

    assert n_threads == 1 or chunk_size[0] == 1, 'Multiprocessing is not safe when chunks are wider than a slice!\n' \
                                                 'Either use n_threads=1 or chunk_size=(1, y, x)'

    from squirrel.library.io import load_data_handle, load_data_from_handle_stack
    from squirrel.library.ome_zarr import create_ome_zarr

    if name is None:
        # This is not entirely correct but who puts a file extension like ome.zarr multiple times into the filename?
        name = str.replace(os.path.split(ome_zarr_filepath)[1], '.ome.zarr', '')

    data_h, shape_h = load_data_handle(stack_path, key=stack_key, pattern=stack_pattern)
    if not append:
        create_ome_zarr(
            ome_zarr_filepath,
            shape=shape_h,
            resolution=resolution,
            unit=unit,
            downsample_type=downsample_type,
            downsample_factors=downsample_factors,
            chunk_size=chunk_size,
            dtype=load_data_from_handle_stack(data_h, 0).dtype,
            name=name
        )
    if z_range is None:
        z_range = [0, shape_h[0]]
    else:
        z_range[1] = min(z_range[1], shape_h[0])

    from squirrel.library.ome_zarr import process_slice_to_ome_zarr
    process_slice_to_ome_zarr(
        stack_path,
        z_range,
        ome_zarr_filepath,
        stack_key=stack_key,
        stack_pattern=stack_pattern,
        save_bounds=save_bounds,
        n_threads=n_threads,
        verbose=verbose
    )

    if downsample_type == 'Average':
        order = 1
    elif downsample_type == 'Sample':
        order = 0
    else:
        raise ValueError(f'Invalid downsample_type = {downsample_type}')

    z_range = np.array(z_range)
    for idx, downsample_factor in enumerate(downsample_factors):
        z_range = (z_range / downsample_factor).astype(int)
        from squirrel.library.ome_zarr import compute_downsampling_layer
        compute_downsampling_layer(
            ome_zarr_filepath,
            z_range,
            f's{idx}', f's{idx + 1}',
            downsample_factor=downsample_factor,
            n_threads=n_threads,
            downsample_order=order,
            verbose=verbose
        )

