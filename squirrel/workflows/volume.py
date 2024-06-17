
import numpy as np


def crop_from_stack_workflow(
        stack_path,
        out_path,
        roi,
        key='data',
        pattern='*.tif',
        verbose=False
):

    if verbose:
        print(f'stack_path = {stack_path}')
        print(f'roi = {roi}')
        print(f'out_path = {out_path}')
        print(f'key = {key}')
        print(f'pattern = {pattern}')

    from squirrel.library.io import load_data_handle, get_filetype

    h, s = load_data_handle(stack_path, key=key, pattern=pattern)

    roi = np.s_[roi[0]: roi[0] + roi[3], roi[1]: roi[1] + roi[4], roi[2]: roi[2] + roi[5]]
    data = h[roi]

    ft_out = get_filetype(out_path)

    if ft_out == 'dir':
        from squirrel.library.io import write_tif_stack
        write_tif_stack(data, out_path)
        return
    if ft_out == 'h5':
        from squirrel.library.io import write_h5_container
        write_h5_container(out_path, data)
        return
    raise ValueError(f'Invalid output type = {ft_out}')

