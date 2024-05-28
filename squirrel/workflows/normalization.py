import numpy as np


def normalize_slices_workflow(
        in_path,
        out_path,
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
        print(f'pattern = {pattern}')
        print(f'in_h5_key = {in_h5_key}')
        print(f'out_h5_key = {out_h5_key}')

    from squirrel.library.io import load_data_handle, write_stack
    from ..library.data import norm_z_range
    stack_handle, stack_shape = load_data_handle(in_path, in_key, in_pattern)
    # z_range = norm_z_range(z_range, stack_shape[0])

    from squirrel.library.normalization import normalize_slices

    # normalized_stack = normalize_slices(stack_handle[z_range[0]: z_range[1]])
    normalized_stack = normalize_slices(stack_handle, z_range=z_range, n_workers=n_workers)
    write_stack(out_path, normalized_stack, key=out_key)
