
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
    from squirrel.library.io import TiffStack

    h, s = load_data_handle(stack_path, key=key, pattern=pattern)

    roi = np.s_[roi[0]: roi[0] + roi[3], roi[1]: roi[1] + roi[4], roi[2]: roi[2] + roi[5]]
    if isinstance(h, TiffStack):
        data = h[:][roi]
    else:
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


def stack_calculator_workflow(
        stack_paths,
        out_path,
        keys=('data', 'data'),
        patterns=('*.tif', '*.tif'),
        operation='add',
        n_workers=1,
        verbose=False
):

    if verbose:
        print(f'stack_paths = {stack_paths}')
        print(f'out_path = {out_path}')
        print(f'keys = {keys}')
        print(f'patterns = {patterns}')
        print(f'operation = {operation}')

    from squirrel.library.io import load_data_handle, get_filetype

    h0, s0 = load_data_handle(stack_paths[0], key=keys[0], pattern=patterns[0])
    h1, s1 = load_data_handle(stack_paths[1], key=keys[1], pattern=patterns[0])

    assert s0 == s1, 'Both stacks must have equal sizes in all three dimensions!'

    from squirrel.library.volume import stack_calculator
    result = stack_calculator(h0, h1, operation=operation, n_workers=n_workers, verbose=verbose)

    ft_out = get_filetype(out_path)

    if ft_out == 'dir':
        from squirrel.library.io import write_tif_stack
        write_tif_stack(result, out_path)
        return
    if ft_out == 'h5':
        from squirrel.library.io import write_h5_container
        write_h5_container(out_path, result)
        return
    raise ValueError(f'Invalid output type = {ft_out}')


if __name__ == '__main__':

    stack_calculator_workflow(
        ('/media/julian/Data/projects/walter/cryo_fib_preprocessing/2024-03-21_2h/InLensCombined',
         '/media/julian/Data/projects/walter/cryo_fib_preprocessing/2024-03-21_2h/InLensCombined'),
        '/tmp/test_stack_calculator',
        operation='average',
        n_workers=16,
        verbose=True
    )
