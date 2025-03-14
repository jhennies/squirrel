import numpy as np


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
        print(f'out_key = {out_key}')
        print(f'z_range = {z_range}')
        print(f'n_workers = {n_workers}')

    from squirrel.library.io import load_data_handle, write_stack
    # from ..library.data import norm_z_range
    stack_handle, stack_shape = load_data_handle(in_path, in_key, in_pattern)

    from squirrel.library.normalization import clahe_on_slices

    normalized_stack = clahe_on_slices(
        stack_handle,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
        z_range=z_range,
        n_workers=n_workers
    )
    write_stack(out_path, normalized_stack, key=out_key)
