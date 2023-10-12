import os

from .io import make_directory, load_h5_container, write_tif_stack
from .io import get_file_list, read_tif_slice, write_tif_slice


def h5_to_tif(
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


def mib_to_tif(
        mib_model_file,
        out_folder,
        verbose=False
):
    if verbose:
        print(f'mib_model_file = {mib_model_file}')
        print(f'out_folder = {out_folder}')

    h5_to_tif(
        mib_model_file,
        'mibModel',
        out_folder,
        axes_order='zxy',
        verbose=verbose
    )


def compress_tif_stack(
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


def merge_tif_stacks(
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
