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
        inconsistent_shapes=False,
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
            if inconsistent_shapes:
                print('Checking for inconsistent shapes')
                this_shapes = []
                st = h[:]
                if type(st) is list:
                    print('Fixing inconsistent shapes')
                    for idx, sl in enumerate(st):
                        print(f'idx = {idx}')
                        this_shapes.append(sl.shape)
                    shapes.append(np.max(this_shapes, axis=0))
                else:
                    print('Shape is consistent')
                    shapes.append(st[0].shape)
                del st
            else:
                shapes.append(s[1:])  # only y and x
            handles.append(h)
        new_shape = np.max(shapes, axis=0)
        if verbose:
            print(f'new_shape = {new_shape}')
        out_idx = 0
        for h in handles:
            if verbose:
                print(f'this_shape = {h[0].shape}')
            # for img in h[:]:
            for idx in range(len(h)):
                img = h[idx]
                print(f'out_idx = {out_idx}')
                write_tif_slice(
                    image_to_shape(img, new_shape),
                    out_folder,
                    out_pattern.format(out_idx)
                )
                out_idx += 1

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


def n5_to_stack_workflow(
        n5_filepath,
        target_dirpath,
        n5_key='setup0/timepoint0/s0',
        z_range=None,
        z_batch_size=None,
        n_threads=1,
        verbose=False
):

    if verbose:
        print(f'n5_filepath = {n5_filepath}')
        print(f'target_dirpath = {target_dirpath}')
        print(f'n5_key = {n5_key}')
        print(f'z_range = {z_range}')
        print(f'n_threads = {n_threads}')

    from squirrel.library.n5 import get_n5_handle

    if z_batch_size is not None:
        assert z_range is None

        handle = get_n5_handle(n5_filepath, n5_key, mode='r')

        for idx in range(0, handle.shape[0], z_batch_size):
            n5_to_stack_workflow(
                n5_filepath,
                target_dirpath,
                n5_key=n5_key,
                z_range=[idx, idx + z_batch_size],
                z_batch_size=None,
                n_threads=n_threads,
                verbose=verbose
            )

        return

    # Load the n5
    from squirrel.library.data import norm_z_range
    handle = get_n5_handle(n5_filepath, n5_key, mode='r')
    z_range = norm_z_range(z_range, handle.shape[0])
    chunk_data = handle[z_range[0]: z_range[1], :]

    # Save to the tif stack
    if not os.path.exists(target_dirpath):
        os.mkdir(target_dirpath)
    from squirrel.library.io import write_tif_stack
    write_tif_stack(chunk_data, target_dirpath, id_offset=z_range[0], slice_name='slice_{:05d}.tif')


def _relabel_and_write_subvolume(
        data_h, z_range, mapping, target_dtype, target_path,
        check_for_existing=True, n_workers=1
):

    if check_for_existing:
        exist_all = True
        for idx in range(*z_range):
            if not os.path.exists(os.path.join(target_path, 'slice{:05d}.tif')):
                exist_all = False
                break
        if exist_all:
            return

    start_idx = z_range[0]
    map_func = np.vectorize(mapping.get)

    relabeled = np.zeros((z_range[1] - z_range[0], data_h.shape[1], data_h.shape[2]), dtype=target_dtype)

    for idy in range(0, data_h.shape[1], data_h.chunks[1]):
        for idx in range(0, data_h.shape[2], data_h.chunks[2]):
            print(f'writing chunk idx, idy, idz: {idx}, {idy}, {z_range[0]}')
            try:
                data = data_h[
                       z_range[0]: z_range[1],
                       idy: idy + data_h.chunks[1],
                       idx: idx + data_h.chunks[2],
                ]
            except TypeError:
                # For tiff slices it has to be done like this
                data = data_h[
                    z_range[0]: z_range[1]
                ][
                    idy: idy + data_h.chunks[1],
                    idx: idx + data_h.chunks[2],
                ]

            # data = data_h[z_range[0]: z_range[1]]
            this_relabeled = map_func(data).astype(target_dtype)
            # print(np.unique(this_relabeled))
            relabeled[:, idy: idy + data_h.chunks[1], idx: idx + data_h.chunks[2]] = this_relabeled

    from squirrel.library.io import write_tif_stack
    write_tif_stack(relabeled, target_path, id_offset=start_idx, slice_name='slice{:05d}.tif')


def cast_dtype_workflow(
        input_path,
        target_path,
        label_mapping=None,
        key=None,
        pattern=None,
        target_key=None,
        target_dtype='uint8',
        z_batch_size=1,
        n_workers=1,
        verbose=False
):

    if verbose:
        print(f'input_path = {input_path}')
        print(f'target_path = {target_path}')
        print(f'type(label_mapping) = {type(label_mapping)}')
        print(f'key = {key}')
        print(f'target_key = {target_key}')
        print(f'target_dtype = {target_dtype}')
        print(f'z_batch_size = {z_batch_size}')
        print(f'n_workers = {n_workers}')

    if label_mapping is not None:
        if type(label_mapping) == str:
            import json
            label_mapping = json.load(open(label_mapping, mode='r'))
        assert type(label_mapping) == dict

    from squirrel.library.io import load_data_handle, get_filetype
    h, shape = load_data_handle(input_path, key, pattern)

    ft = get_filetype(target_path)
    if ft != 'dir':
        raise ValueError('Only implemented for tif stack output')

    if not os.path.exists(target_path):
        os.mkdir(target_path)

    if n_workers == 1:
        for idx in range(0, shape[0], z_batch_size):
            z_range = [idx, min(idx + z_batch_size, shape[0])]
            _relabel_and_write_subvolume(h, z_range, label_mapping, target_dtype, target_path)
    else:
        # from concurrent.futures import ThreadPoolExecutor
        # with ThreadPoolExecutor(max_workers=n_workers) as tpe:
        #     tasks = [
        #         tpe.submit(
        #             _relabel_and_write_subvolume,
        #             h, [idx, min(idx + z_batch_size, shape[0])], label_mapping, target_dtype, target_path
        #         )
        #         for idx in range(0, shape[0], z_batch_size)
        #     ]
        #     [task.result() for task in tasks]

        from multiprocessing import Pool
        with Pool(processes=n_workers) as p:
            tasks = [
                p.apply_async(
                    _relabel_and_write_subvolume,
                    (h, [idx, min(idx + z_batch_size, shape[0])], label_mapping, target_dtype, target_path))
                for idx in range(0, shape[0], z_batch_size)
            ]
            [task.get() for task in tasks]

    # for idx in range(0, shape[0], z_batch_size):
    #     z_range = [idx, min(idx + z_batch_size, shape[0])]
    #     _relabel_and_write_subvolume(h, z_range, label_mapping, target_dtype, target_path, n_workers=n_workers)


def cast_segmentation_workflow(
        input_path,
        target_path,
        key=None,  # Defaults: ome.zarr: "s0"; n5: "setup0/timepoint0/s0"
        pattern=None,
        target_key=None,  # Defaults to input_key
        target_dtype=None,  # None: Tries to find the best-suitable data type. Only for integer types and only use for segmentations!
        out_json=None,
        z_batch_size=1,
        n_workers=1,
        verbose=False
):

    from squirrel.library.io import load_data_handle
    from squirrel.workflows.volume import get_label_list_workflow
    from squirrel.library.data import get_optimal_dtype

    # If target_dtype is None and source_dtype in ['uint8', 'uint16', 'uint32', 'uint64']
    #   -> Determine all labels in the data
    label_mapping = None

    if out_json is not None:
        assert target_dtype is None
        if os.path.exists(out_json):
            if verbose:
                print(f'Reading out json: {out_json} ...')
            import json
            label_list = json.load(open(out_json, mode='r'))
            label_mapping = dict(zip(label_list, range(len(label_list))))
            target_dtype = get_optimal_dtype(len(label_list))

    if target_dtype is None and label_mapping is None:

        if verbose:
            print(f'Computing label list and mapping ...')

        h, shape = load_data_handle(input_path, key, pattern)

        if h.dtype in ['uint8', 'uint16', 'uint32', 'uint64']:
            label_list = get_label_list_workflow(
                input_path, key, pattern,
                z_batch_size=z_batch_size, n_workers=n_workers,
                out_json=out_json,
                quiet=True, verbose=verbose
            )

            # Set up a dictionary to re-assign the labels
            label_mapping = dict(zip(label_list, range(len(label_list))))

            if verbose:
                print(f'label_mapping = {label_mapping}')

            target_dtype = get_optimal_dtype(len(label_list))

        else:
            raise ValueError('target_dtype needs to be specified for non-integer volumes')

    # Iterate the batches and write the results
    cast_dtype_workflow(
        input_path,
        target_path,
        label_mapping=label_mapping,
        key=key,
        pattern=pattern,
        target_key=target_key,
        target_dtype=target_dtype,
        z_batch_size=z_batch_size,
        n_workers=n_workers,
        verbose=verbose
    )
