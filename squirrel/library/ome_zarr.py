
import numpy as np


def create_ome_zarr(
        filepath,
        shape,
        resolution=(1., 1., 1.),
        unit='pixel',
        downsample_type='Average',  # One of ['Average', 'Sample']
        downsample_factors=(2, 2, 2),
        chunk_size=(1, 256, 256),
        dtype='uint8',
        name=None
):

    from zarr import open as zarr_open
    handle = zarr_open(filepath, mode='w')
    handle.create_dataset(
        's0',
        shape=shape,
        compression='gzip',
        chunks=chunk_size,
        dtype=dtype,
        dimension_separator='/'
    )

    if name == None:
        import os
        name = str.replace(os.path.split(filepath)[1], '.ome.zarr', '')

    def dataset(path, scale):
        return dict(
            path=path,
            coordinateTransformations=[
                dict(
                    type='scale',
                    scale=scale
                )
            ]
        )

    def axes():
        return [
            dict(
                name=axis,
                type='space',
                unit=unit
            )
            for axis in 'zyx'
        ]

    handle.attrs.update(
        dict(
            multiscales=[
                dict(
                    axes=axes(),
                    datasets=[dataset('s0', list(resolution))],
                    name=name,
                    type=downsample_type,
                    version="0.4"
                )
            ]
        )
    )

    scale = 1
    s_idx = 0
    for downsample_factor in downsample_factors:

        scale *= downsample_factor
        s_idx += 1

        handle.create_dataset(
            f's{s_idx}',
            shape=(np.array(shape) / scale).astype(int).tolist(),
            compression='gzip',
            chunks=chunk_size,
            dtype=dtype,
            dimension_separator='/'
        )

        attrs = handle.attrs
        attrs['multiscales'][0]['datasets'].append(
            dataset(f's{s_idx}', list(np.array(resolution) * scale))
        )

    handle.attrs.update(attrs)


def get_ome_zarr_handle(
        filepath,
        key=None,
        mode='r'
):
    from zarr import open as zarr_open
    if key is not None:
        return zarr_open(filepath, mode=mode)[key]
    return zarr_open(filepath, mode=mode)


def slice_to_ome_zarr(
        stack_path,
        slice_idx,
        ome_zarr_handle,
        stack_key='data',
        stack_pattern='*.tif',
        save_bounds=False,
        verbose=False
):
    print(f'slice_idx = {slice_idx}')
    if verbose:
        print(f'stack_path = {stack_path}')
        print(f'ome_zarr_handle = {ome_zarr_handle}')
        print(f'save_bounds = {save_bounds}')

    from .io import load_data_from_handle_stack, load_data_handle

    data_handle, shape_h = load_data_handle(stack_path, key=stack_key, pattern=stack_pattern)
    slice_data, _ = load_data_from_handle_stack(data_handle, slice_idx)
    print(f'loaded slice data...')

    if save_bounds:
        # TODO this
        pass

    ome_zarr_handle[slice_idx, :] = slice_data
    print(f'slice data written')


def process_slice_to_ome_zarr(
        stack_path,
        z_range,
        ome_zarr_filepath,
        stack_key='data',
        stack_pattern='*.tif',
        save_bounds=False,
        n_threads=1,
        verbose=False
):

    ome_zarr_h = get_ome_zarr_handle(ome_zarr_filepath, 's0', 'a')

    if n_threads == 1:

        for idx in range(*z_range):
            slice_to_ome_zarr(
                stack_path,
                idx,
                ome_zarr_h,
                stack_key=stack_key,
                stack_pattern=stack_pattern,
                save_bounds=save_bounds,
                verbose=verbose
            )

    else:

        from multiprocessing import Pool
        with Pool(processes=n_threads) as p:
            tasks = [
                p.apply_async(slice_to_ome_zarr, (
                    stack_path,
                    idx,
                    ome_zarr_h,
                    stack_key,
                    stack_pattern,
                    save_bounds,
                    verbose
                ))
                for idx in range(*z_range)
            ]
            [task.get() for task in tasks]


def slice_of_downsampling_layer(
        source_ome_zarr_handle,
        target_ome_zarr_handle,
        # source_slice_idx,
        target_slice_idx,
        downsample_factor=2,
        downsample_order=1,
        verbose=False
):

    print(f'target_slice_idx = {target_slice_idx}')
    if verbose:
        print(f'source_ome_zarr_handle = {source_ome_zarr_handle}')
        print(f'target_ome_zarr_handle = {target_ome_zarr_handle}')
        print(f'downsample_factor = {downsample_factor}')

    from .transformation import apply_affine_transform, setup_scale_matrix, validate_and_reshape_matrix

    source_slice_idx = target_slice_idx * downsample_factor
    source_data = source_ome_zarr_handle[source_slice_idx: source_slice_idx + downsample_factor]
    if downsample_order == 0:
        source_data = source_data[int(downsample_factor / 2) - 1, :]
    elif downsample_order == 1:
        source_data = np.mean(source_data, axis=0)
    else:
        raise NotImplementedError(f'Downsample order = {downsample_order} is not implemented!')
    transform_matrix = validate_and_reshape_matrix(
        setup_scale_matrix([downsample_factor] * 2, ndim=2),
        ndim=2
    )
    target_data = apply_affine_transform(
        source_data,
        transform_matrix,
        order=downsample_order,
        scale_canvas=True,
        no_offset_to_center=True,
        verbose=verbose
    )
    # # For a reason that I currently don't understand this does not work
    # #   Regardless of downsample_order is always produces as if it was downsample_order=0
    # transform_matrix = validate_and_reshape_matrix(
    #     setup_scale_matrix([downsample_factor] * 3, ndim=3),
    #     ndim=3
    # )
    # target_data = apply_affine_transform(
    #     source_data,
    #     transform_matrix,
    #     order=downsample_order,
    #     scale_canvas=True,
    #     no_offset_to_center=True,
    #     verbose=verbose
    # )

    # this_slice_idx = int(source_slice_idx / downsample_factor)
    this_slice_idx = target_slice_idx
    target_data = target_data.squeeze()
    try:
        target_ome_zarr_handle[this_slice_idx, :] = target_data
    except ValueError:
        # This happens if, due to downscaling, the target slice is one pixel larger than the ome-zarr dataset
        this_shape = target_ome_zarr_handle[this_slice_idx].shape
        target_ome_zarr_handle[this_slice_idx, :] = target_data[:this_shape[0], :this_shape[1]]


def compute_downsampling_layer(
        ome_zarr_filepath,
        z_range,
        source_layer,
        target_layer,
        downsample_factor=2,
        downsample_order=1,
        n_threads=1,
        verbose=False
):

    source_h = get_ome_zarr_handle(ome_zarr_filepath, source_layer, 'r')
    target_h = get_ome_zarr_handle(ome_zarr_filepath, target_layer, 'a')

    if n_threads == 1:

        for idx in range(*z_range, downsample_factor):
            slice_of_downsampling_layer(
                source_h,
                target_h,
                idx,
                downsample_factor,
                downsample_order=downsample_order,
                verbose=verbose
            )
    else:

        from multiprocessing import Pool
        with Pool(processes=n_threads) as p:
            tasks = [
                p.apply_async(slice_of_downsampling_layer, (
                    source_h,
                    target_h,
                    idx,
                    downsample_factor,
                    downsample_order,
                    verbose
                ))
                for idx in range(*z_range)
            ]
            [task.get() for task in tasks]


def get_scale_of_downsample_level(handle, downsample_level):

    datasets = handle.attrs['multiscales'][0]['datasets']
    this_dataset = datasets[downsample_level]
    this_path = this_dataset['path']
    assert this_path == f's{downsample_level}', \
        f'Invalid path to downsample level combination: {this_path} != s{downsample_level}'

    return this_dataset['coordinateTransformations'][0]['scale']
