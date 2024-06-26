
import numpy as np
import os


# NOTE: deprecated
# def _load_data(filepath, key='data'):
#
#     from ..library.io import get_filetype
#
#     if get_filetype(filepath) == 'h5':
#         from ..library.io import load_h5_container
#         return load_h5_container(filepath, key)
#     if get_filetype(filepath) == 'nii':
#         from ..library.io import load_nii_file
#         return load_nii_file(filepath)


def elastix3d(
        moving_filepath,
        fixed_filepath,
        out_filepath,
        moving_key='data',
        fixed_key='data',
        transform='affine',
        automatic_transform_initialization=False,
        pivot=(0., 0., 0.),
        view_results_in_napari=False,
        verbose=False
):

    from ..library.elastix import register_with_elastix

    fixed_image = _load_data(fixed_filepath, key=fixed_key)
    moving_image = _load_data(moving_filepath, key=moving_key)

    if verbose:
        print(f'type(fixed_image) = {type(fixed_image)}')
        print(f'type(moving_image) = {type(moving_image)}')

    elastix_result = register_with_elastix(
        fixed_image, moving_image,
        transform=transform,
        automatic_transform_initialization=automatic_transform_initialization,
        verbose=verbose
    )

    result_transform = elastix_result['affine_parameters']
    result_image = elastix_result['result_image']

    from ..library.io import write_h5_container
    write_h5_container(os.path.join(out_filepath), result_image, 'data')

    from ..library.elastix import save_transforms
    save_transforms(
        result_transform,
        os.path.join(
            os.path.split(out_filepath)[0],
            os.path.splitext(os.path.split(out_filepath)[1])[0] + '.json'
        ),
        param_order='elastix',
        save_order='C',
        ndim=3,
        verbose=verbose
    )

    if view_results_in_napari:
        from ..workflows.viewing import view_in_napari
        view_in_napari(
            images=[moving_image, fixed_image, result_image],
            image_keys=['moving', 'fixed', 'result'],
            verbose=verbose
        )


def register_z_chunks(
        moving_filepath,
        fixed_filepath,
        out_path,
        moving_key='data',
        fixed_key='data',
        z_chunk_size=16,
        transform='affine',
        automatic_transform_initialization=False,
        view_results_in_napari=False,
        verbose=False
):

    from ..library.elastix import register_with_elastix, save_transforms
    from ..library.io import write_h5_container
    from ..library.io import make_directory

    make_directory(out_path, exist_ok=True)

    fixed_image = _load_data(fixed_filepath, key=fixed_key)
    moving_image = _load_data(moving_filepath, key=moving_key)

    def _cast_to_uint8(image):
        if image.dtype != 'uint8':
            print(f'Warning: Input image was not uint8. Casting the filetype, however this may cause unintended effects')
            return image.astype('uint8')
        return image

    fixed_image = _cast_to_uint8(fixed_image)
    moving_image = _cast_to_uint8(moving_image)
    result_images = []
    result_transforms = []
    result_affine_transforms = []

    for chunk_start in range(0, fixed_image.shape[0], z_chunk_size):

        out_images_filepath = os.path.join(out_path, 'chunk_{:05d}.h5'.format(chunk_start))

        this_fixed = fixed_image[chunk_start: chunk_start + z_chunk_size]
        this_moving = moving_image[chunk_start: chunk_start + z_chunk_size]

        if this_fixed.shape[0] < z_chunk_size:
            break
        if this_moving.shape[0] < z_chunk_size:
            break
        if verbose:
            print(f'this_fixed.shape = {this_fixed.shape}')
            print(f'this_moving.shape = {this_moving.shape}')

        write_h5_container(out_images_filepath, this_fixed, key='fixed')
        write_h5_container(out_images_filepath, this_moving, key='moving', append=True)

        this_result_dict = register_with_elastix(
            this_fixed, this_moving,
            out_dir=out_path,
            transform=transform,
            automatic_transform_initialization=automatic_transform_initialization,
            verbose=verbose
        )
        result_images.append(this_result_dict['result_image'])
        result_affine_transforms.append(this_result_dict['affine_parameters'])
        if transform == 'rigid':
            result_transforms.append([float(x) for x in this_result_dict['rigid_parameters']])
        if transform == 'SimilarityTransform':
            result_transforms.append([float(x) for x in this_result_dict['similarity_parameters']])

        write_h5_container(out_images_filepath, result_images[-1], 'result', append=True)

        if verbose:
            print(f'Done with chunk {chunk_start}')

    if verbose:
        print(f'Saving transform to {os.path.join(out_path, "affine_transforms.json")}')
    save_transforms(
        result_affine_transforms, os.path.join(out_path, 'affine_transforms.json'),
        param_order='elastix', save_order='C', ndim=3, verbose=verbose)
    import json
    with open(os.path.join(out_path, 'transforms.json'), mode='w') as f:
        json.dump(result_transforms, f, indent=2)

    if verbose:
        print(f'len(result_images) = {len(result_images)}')
        print(f'result_images[0].shape = {result_images[0].shape}')
    result_image = np.concatenate(result_images, axis=0)
    if verbose:
        print(f'result_image.shape = {result_image.shape}')

    write_h5_container(os.path.join(out_path, 'result.h5'), result_image, key='data')

    if view_results_in_napari:
        from ..workflows.viewing import view_in_napari
        view_in_napari(
            images=[moving_image, fixed_image, result_image],
            image_names=['moving', 'fixed', 'result'],
            invert_images=True,
            verbose=verbose
        )


def slices_to_volume(*args, **kwargs):
    raise RuntimeError('This function was renamed, please use "slice_wise_stack_to_stack_alignment_workflow!"')


def slice_wise_stack_to_stack_alignment_workflow(
        moving_path,
        fixed_path,
        out_filepath,
        moving_key='data',
        fixed_key='data',
        moving_pattern='*.tif',
        fixed_pattern='*.tif',
        transform='affine',
        automatic_transform_initialization=False,
        auto_mask=False,
        number_of_spatial_samples=None,
        maximum_number_of_iterations=None,
        number_of_resolutions=None,
        pre_fix_big_jumps=False,
        z_range=None,
        verbose=False
):

    from ..library.io import write_h5_container, make_directory, load_data_handle
    from ..library.elastix import slice_wise_stack_to_stack_alignment

    out_basename = os.path.splitext(out_filepath)[0]

    elastix_cache = out_basename + '.elastix_cache'
    make_directory(elastix_cache, exist_ok=True)

    fixed_handle, fixed_shape = load_data_handle(fixed_path, fixed_key, fixed_pattern)
    moving_handle, moving_shape = load_data_handle(moving_path, moving_key, moving_pattern)
    fixed_volume = fixed_handle[z_range[0]: z_range[1]]
    moving_volume = moving_handle[z_range[0]: z_range[1]]

    def _cast_to_uint8(image):
        if image.dtype != 'uint8':
            print(f'Warning: Input image was not uint8. Casting the filetype, however this may cause unintended effects')
            return image.astype('uint8')
        return image

    fixed_volume = _cast_to_uint8(fixed_volume)
    moving_volume = _cast_to_uint8(moving_volume)

    assert fixed_volume.shape[0] >= moving_volume.shape[0]

    result_transforms, result_volume = slice_wise_stack_to_stack_alignment(
        moving_volume,
        fixed_volume,
        transform=transform,
        automatic_transform_initialization=automatic_transform_initialization,
        out_dir=elastix_cache,
        auto_mask=auto_mask,
        number_of_spatial_samples=number_of_spatial_samples,
        maximum_number_of_iterations=maximum_number_of_iterations,
        number_of_resolutions=number_of_resolutions,
        return_result_image=True,
        pre_fix_big_jumps=pre_fix_big_jumps,
        verbose=verbose
    )

    result_transforms.to_file(out_basename + '.json')
    write_h5_container(out_filepath, np.array(result_volume), key='data', append=False)


def elastix_stack_alignment_workflow(
        stack,
        out_filepath,
        transform='translation',
        key='data',
        pattern='*.tif',
        auto_mask=False,
        number_of_spatial_samples=None,
        maximum_number_of_iterations=None,
        number_of_resolutions=None,
        pre_fix_big_jumps=False,
        pre_fix_iou_thresh=0.5,
        gaussian_sigma=0.,
        use_clahe=False,
        use_edges=False,
        z_range=None,
        z_step=1,
        average_for_z_step=False,
        determine_bounds=False,
        parameter_map=None,
        quiet=False,
        overwrite=False,
        verbose=False
):

    if not overwrite and os.path.exists(out_filepath):
        print(f'Target file exists: {out_filepath}\nSkipping elastix stack alignment workflow ...')
        return

    from ..library.io import load_data_handle
    from ..library.elastix import register_with_elastix
    from ..library.affine_matrices import AffineMatrix, AffineStack
    from ..library.data import norm_z_range

    stack, stack_size = load_data_handle(stack, key=key, pattern=pattern)

    transforms = AffineStack(is_sequenced=False, pivot=[0., 0.])
    bounds = []

    z_range = norm_z_range(z_range, stack_size[0])

    for idx in range(*z_range, z_step):
        if not quiet:
            print(f'idx = {idx} / {z_range[1]}')

        if average_for_z_step:
            z_slice_moving = np.mean(stack[idx: idx + z_step], axis=0).astype('uint8')
        else:
            z_slice_moving = stack[idx]

        if idx == 0:
            transforms.append(AffineMatrix([1., 0., 0., 0., 1., 0.], pivot=[0., 0.]))
        else:

            if average_for_z_step:

                z_slice_fixed = np.mean(stack[idx - z_step: idx], axis=0).astype('uint8')
            else:

                z_slice_fixed = stack[idx - z_step]

            result_matrix, _ = register_with_elastix(
                z_slice_fixed,
                z_slice_moving,
                transform=transform,
                automatic_transform_initialization=False,
                auto_mask=auto_mask,
                number_of_spatial_samples=number_of_spatial_samples,
                maximum_number_of_iterations=maximum_number_of_iterations,
                number_of_resolutions=number_of_resolutions,
                pre_fix_big_jumps=pre_fix_big_jumps,
                pre_fix_iou_thresh=pre_fix_iou_thresh,
                parameter_map=parameter_map,
                return_result_image=False,
                params_to_origin=True,
                gaussian_sigma=gaussian_sigma,
                use_clahe=use_clahe,
                use_edges=use_edges,
                verbose=verbose
            )
            transforms.append(result_matrix)

        # FIXME: This is already performed inside register_with_elastix if auto_mask is on
        if determine_bounds:
            from ..library.image import get_bounds
            bounds.append(get_bounds(z_slice_moving, return_ints=True))

    if z_step > 1:
        # transforms = transforms.get_sequenced_stack()
        # transforms = transforms.get_interpolated(z_step)
        transforms.set_meta('z_step', z_step)

    if determine_bounds:
        assert len(transforms) == len(bounds)
        transforms.set_meta('bounds', np.array(bounds))
    transforms.to_file(out_filepath)


def stack_alignment_validation_workflow(
        stack,
        out_dirpath,
        rois,
        key='data',
        pattern='*.tif',
        resolution_yx=(1.0, 1.0),
        out_name=None,
        y_max=None,
        verbose=False
):

    if verbose:
        print(f'stack = {stack}')
        print(f'out_dirpath = {out_dirpath}')
        print(f'rois = {rois}')
        print(f'key = {key}')
        print(f'pattern = {pattern}')
        print(f'resolution_yx = {resolution_yx}')

    from squirrel.library.io import load_data_handle
    from squirrel.library.elastix import register_with_elastix
    from squirrel.library.affine_matrices import AffineStack, AffineMatrix
    from squirrel.library.transformation import apply_stack_alignment
    from matplotlib import pyplot as plt
    from h5py import File

    if not os.path.exists(out_dirpath):
        os.mkdir(out_dirpath)

    if out_name is None:
        transforms_filepath = os.path.join(out_dirpath, 'transforms.json')
        plot_filepath = os.path.join(out_dirpath, 'plot.pdf')
        image_dirpath = os.path.join(out_dirpath, 'images')
    else:
        transforms_filepath = os.path.join(out_dirpath, f'transforms-{out_name}.json')
        plot_filepath = os.path.join(out_dirpath, f'plot-{out_name}.pdf')
        image_dirpath = os.path.join(out_dirpath, f'images-{out_name}')
    if not os.path.exists(image_dirpath):
        os.mkdir(image_dirpath)
    image_filepath = os.path.join(image_dirpath, 'image_{:04d}.h5')

    stack, stack_size = load_data_handle(stack, key=key, pattern=pattern)
    labels = []

    for roi_idx, roi in enumerate(rois):

        print(f'roi_idx = {roi_idx} / {len(rois) - 1}')

        if verbose:
            print(f'roi = {roi}')
        roi_data = stack[:][roi]
        transforms = AffineStack(is_sequenced=False, pivot=[0., 0.])
        transforms.append(AffineMatrix(parameters=[1., 0., 0., 0., 1., 0.]))

        for idx in range(len(roi_data) - 1):

            # print(f'idx = {idx} / {len(roi_data) - 2}')

            z_slice_fixed = roi_data[idx]
            z_slice_moving = roi_data[idx + 1]

            result_matrix, _ = register_with_elastix(
                z_slice_fixed,
                z_slice_moving,
                transform='translation',
                automatic_transform_initialization=False,
                auto_mask=False,
                # number_of_spatial_samples=256,
                # maximum_number_of_iterations=256,
                # number_of_resolutions=1,
                pre_fix_big_jumps=False,
                return_result_image=True,
                params_to_origin=True,
                gaussian_sigma=2.0,
                verbose=verbose
            )
            transforms.append(result_matrix)

            # result_volume.append(result_image)

        result_volume = apply_stack_alignment(
            roi_data,
            roi_data.shape,
            transforms,
            n_workers=1,
            verbose=verbose
        )
        with File(image_filepath.format(roi_idx), mode='w') as f:
            f.create_dataset('data', data=result_volume, compression='gzip')

        translations = np.array(transforms.get_translations()) * resolution_yx
        transforms.to_file(transforms_filepath)
        labels.append(f'roi-{roi_idx}')
        plt.plot(np.sqrt(translations[:, 0] ** 2 + translations[:, 1] ** 2), label=labels[-1])

    plt.ylim(ymin=0, ymax=y_max)
    plt.legend()
    plt.savefig(plot_filepath)
    # plt.show()


def apply_multi_step_stack_alignment_workflow(
        image_stack,
        transform_paths,
        out_filepath=None,
        key='data',
        pattern='*.tif',
        auto_pad=False,
        target_image_shape=None,
        z_range=None,
        start_transform_id=0,
        n_workers=1,
        quiet=False,
        write_result=False,
        verbose=False,
):
    from squirrel.library.elastix import ElastixMultiStepStack, ElastixStack
    from squirrel.library.affine_matrices import AffineStack

    from squirrel.library.io import load_data_handle
    if target_image_shape is None:
        image_stack_h, stack_shape = load_data_handle(image_stack, key=key, pattern=pattern)
        target_image_shape = stack_shape[1:]
    else:
        assert not auto_pad, "Don't supply a stack shape if auto padding will be performed!"
        image_stack_h, _ = load_data_handle(image_stack, key=key, pattern=pattern)

    stacks = []
    for transform_path in transform_paths:
        if os.path.isdir(transform_path):
            stack = ElastixStack(dirpath=transform_path)  # , image_shape=target_image_shape))
            if z_range is not None:
                stack = ElastixStack(stack=stack[start_transform_id: start_transform_id + z_range[1] - z_range[0]])
            stacks.append(stack)
        else:
            stack = AffineStack(filepath=transform_path)
            if z_range is not None:
                stack = stack.new_stack_with_same_meta(stack[start_transform_id: start_transform_id + z_range[1] - z_range[0]])
            if stack.exists_meta('stack_shape'):
                image_shape = stack.get_meta('stack_shape')[1:]
            else:
                image_shape = target_image_shape
            if not stack.is_sequenced:
                stack = stack.get_sequenced_stack()
            stacks.append(ElastixStack(stack=stack, image_shape=image_shape))

    if verbose:
        print(f'target_image_shape = {target_image_shape}')

    emss = ElastixMultiStepStack(stacks=stacks, image_shape=target_image_shape)

    result_volume = emss.apply_on_image_stack(
        image_stack_h,
        target_image_shape=target_image_shape,
        z_range=z_range,
        n_workers=n_workers,
        quiet=quiet,
        verbose=verbose
    )

    if write_result:
        from squirrel.library.io import write_h5_container
        write_h5_container(out_filepath, result_volume)
        return result_volume
    else:
        return result_volume


if __name__ == '__main__':
    # stack_alignment_validation_workflow(
    #     '/media/julian/Data/projects/kors/align/4T/amst_parameter_test/pre_align/pre-align.ome.zarr',
    #     # '/media/julian/Data/projects/kors/align/4T/amst_parameter_test/amst_results_02/amst-0001-ref.h5',
    #     None,
    #     [
    #         np.s_[:, 330: 458, 2518: 2646],  # Right edge
    #         np.s_[:, 386: 514, 440: 568],  # Left edge
    #         np.s_[:, 650: 778, 1640: 1768]  # Bottom
    #     ],
    #     key='s0',
    #     # key='data',
    #     resolution_yx=[1, 1]
    # )

    apply_multi_step_stack_alignment_workflow(
        '/media/julian/Data/projects/hennies/amst_devel/hela-tm.ome.zarr/',
        ['/media/julian/Data/projects/hennies/amst_devel/amst-elastic-transforms-01/'],
        '/media/julian/Data/projects/hennies/amst_devel/amst-elastic-transforms-01.h5',
        key='s0',
        auto_pad=False,
        target_image_shape=None,
        z_range=None,
        n_workers=1,
        quiet=False,
        verbose=False,
    )


