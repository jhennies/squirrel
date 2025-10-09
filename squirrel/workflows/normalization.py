import os

import numpy as np


def adjust_greyscale_workflow(
        in_path,
        out_path,
        in_pattern='*.tif',
        in_key=None,
        out_key=None,
        greys_in=None,
        greys_out=None,
        dtype_out='uint8',
        n_workers=os.cpu_count(),
        verbose=False
):

    if verbose:
        print(f'in_path = {in_path}')
        print(f'out_path = {out_path}')
        print(f'in_pattern = {in_pattern}')
        print(f'in_key = {in_key}')
        print(f'out_key = {out_key}')
        print(f'greys_in = {greys_in}')
        print(f'greys_out = {greys_out}')
        print(f'dtype_out = {dtype_out}')
        print(f'n_workers = {n_workers}')

    from squirrel.library.io import load_data_handle, write_stack
    # from ..library.data import norm_z_range
    stack_handle, stack_shape = load_data_handle(in_path, in_key, in_pattern)

    from squirrel.library.normalization import adjust_greyscale
    result = adjust_greyscale(
        stack_handle,
        z_range=None,
        greys_in=greys_in,
        greys_out=greys_out,
        cast_dtype=dtype_out,
        n_workers=n_workers
    )

    write_stack(out_path, result, key=out_key)


def normalize_slices_workflow(
        in_path,
        out_path,
        in_pattern='*.tif',
        in_key='data',
        out_key='data',
        dilate_background=0,
        quantiles=(0.1, 0.9),
        anchors=(0.2, 0.8),
        z_range=None,
        keep_zeros=False,
        n_workers=1,
        verbose=False
):

    if verbose:
        print(f'in_path = {in_path}')
        print(f'out_path = {out_path}')
        print(f'pattern = {in_pattern}')
        print(f'in_h5_key = {in_key}')
        print(f'out_h5_key = {out_key}')

    from squirrel.library.io import load_data_handle, write_stack
    # from ..library.data import norm_z_range
    stack_handle, stack_shape = load_data_handle(in_path, in_key, in_pattern)

    from squirrel.library.normalization import normalize_slices

    normalized_stack = normalize_slices(
        stack_handle,
        dilate_background=dilate_background,
        quantiles=quantiles,
        anchors=anchors,
        keep_zeros=keep_zeros,
        z_range=z_range,
        n_workers=n_workers
    )
    write_stack(out_path, normalized_stack, key=out_key)


def clahe_on_slices_workflow(
        in_path,
        out_path,
        clip_limit=3.0,
        tile_grid_size=(127, 127),
        in_pattern='*.tif',
        in_key='data',
        out_key='data',
        z_range=None,
        cast_dtype=None,
        invert_output=False,
        gaussian_sigma=0.0,
        batch_size=None,
        n_workers=1,
        verbose=False
):

    if verbose:
        print(f'in_path = {in_path}')
        print(f'out_path = {out_path}')
        print(f'clip_limit = {clip_limit}')
        print(f'tile_grid_size = {tile_grid_size}')
        print(f'in_pattern = {in_pattern}')
        print(f'in_key = {in_key}')
        print(f'cast_dtype = {cast_dtype}')
        print(f'invert_output = {invert_output}')
        print(f'gaussian_sigma = {gaussian_sigma}')
        print(f'out_key = {out_key}')
        print(f'z_range = {z_range}')
        print(f'n_workers = {n_workers}')

    from squirrel.library.io import load_data_handle, write_stack, get_filetype
    # from ..library.data import norm_z_range
    stack_handle, stack_shape = load_data_handle(in_path, in_key, in_pattern)

    from squirrel.library.normalization import clahe_on_slices

    if batch_size is None:
        normalized_stack = clahe_on_slices(
            stack_handle,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            cast_dtype=cast_dtype,
            invert_output=invert_output,
            gaussian_sigma=gaussian_sigma,
            z_range=z_range,
            n_workers=n_workers
        )
        write_stack(out_path, normalized_stack, key=out_key)
        return

    assert get_filetype(out_path) == 'dir', 'Batched processing only implemented for tif stack output!'

    from squirrel.library.io import write_tif_stack
    for zidx in range(0, stack_shape[0], batch_size):

        normalized_substack = clahe_on_slices(
            stack_handle,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            cast_dtype=cast_dtype,
            invert_output=invert_output,
            gaussian_sigma=gaussian_sigma,
            z_range=[zidx, zidx + batch_size],
            n_workers=n_workers
        )

        write_tif_stack(normalized_substack, out_path, id_offset=zidx, slice_name='slice_{:05}.tif')
