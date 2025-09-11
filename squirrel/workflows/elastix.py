
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


def register_with_elastix_workflow(
        moving_filepath,
        fixed_filepath,
        out_filepath,
        out_img_filepath=None,
        transform='affine',
        auto_mask=None,
        number_of_spatial_samples=None,
        maximum_number_of_iterations=None,
        number_of_resolutions=None,
        initialize_offsets_method=None,
        initialize_offsets_kwargs=None,
        gaussian_sigma=0.,
        use_clahe=False,
        use_edges=False,
        parameter_map=None,
        debug_dirpath=None,
        n_workers=os.cpu_count(),
        verbose=False
):

    from squirrel.library.elastix import register_with_elastix

    if not os.path.exists(out_filepath):

        transform, _ = register_with_elastix(
            fixed_filepath,
            moving_filepath,
            transform=transform,
            automatic_transform_initialization=False,
            out_dir=None,
            params_to_origin=True,
            auto_mask=auto_mask,
            number_of_spatial_samples=number_of_spatial_samples,
            maximum_number_of_iterations=maximum_number_of_iterations,
            number_of_resolutions=number_of_resolutions,
            return_result_image=False,
            initialize_offsets_method=initialize_offsets_method,
            initialize_offsets_kwargs=initialize_offsets_kwargs,
            parameter_map=parameter_map,
            gaussian_sigma=gaussian_sigma,
            use_edges=use_edges,
            use_clahe=use_clahe,
            crop_to_bounds_off=False,
            n_workers=n_workers,
            normalize_images=True,
            result_to_disk='',
            debug_dirpath=debug_dirpath,
            verbose=verbose
        )

        transform.to_file(out_filepath)

    else:
        from squirrel.library.affine_matrices import AffineMatrix
        transform = AffineMatrix(filepath=out_filepath)

    if out_img_filepath:
        from squirrel.library.elastix import apply_transforms_on_image
        registered_img = apply_transforms_on_image(moving_filepath, [transform], n_workers=n_workers, verbose=verbose)
        from tifffile import imwrite
        imwrite(out_img_filepath, registered_img)


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
        initialize_offsets_method=None,
        initialize_offsets_kwargs=None,
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
        initialize_offsets_method=initialize_offsets_method,
        initialize_offsets_kwargs=initialize_offsets_kwargs,
        verbose=verbose
    )

    result_transforms.to_file(out_basename + '.json')
    write_h5_container(out_filepath, np.array(result_volume), key='data', append=False)


def _elastix_one_slice(
        idx,
        z_slice_moving,
        z_slice_fixed,
        z_range,
        determine_bounds,
        transform,
        auto_mask,
        number_of_spatial_samples,
        maximum_number_of_iterations,
        number_of_resolutions,
        initialize_offsets_method,
        initialize_offsets_kwargs,
        parameter_map,
        gaussian_sigma,
        use_clahe,
        use_edges,
        verbose=False,
        debug=False, out_filepath=None, quiet=False
):

    from squirrel.library.elastix import register_with_elastix
    from squirrel.library.affine_matrices import AffineMatrix

    if not quiet:
        print(f'idx = {idx} / {z_range[1]}')

    if idx == 0:
        result_matrix = AffineMatrix([1., 0., 0., 0., 1., 0.], pivot=[0., 0.])
    else:

        result_matrix, _ = register_with_elastix(
            z_slice_fixed,
            z_slice_moving,
            transform=transform,
            automatic_transform_initialization=False,
            auto_mask=auto_mask,
            number_of_spatial_samples=number_of_spatial_samples,
            maximum_number_of_iterations=maximum_number_of_iterations,
            number_of_resolutions=number_of_resolutions,
            initialize_offsets_method=initialize_offsets_method,
            initialize_offsets_kwargs=initialize_offsets_kwargs,
            parameter_map=parameter_map,
            return_result_image=False,
            params_to_origin=True,
            gaussian_sigma=gaussian_sigma,
            use_clahe=use_clahe,
            use_edges=use_edges,
            verbose=verbose,
            debug_dirpath=None if not debug else '{}.debug/{:05d}'.format(os.path.splitext(out_filepath)[0], idx),
            n_workers=1
        )

    # FIXME: This is already performed inside register_with_elastix if auto_mask is on
    bounds = None
    if determine_bounds:
        from squirrel.library.image import get_bounds
        bounds = get_bounds(z_slice_moving, return_ints=True)

    return result_matrix, bounds


def elastix_stack_alignment_workflow(
        stack,
        out_filepath,
        transform='translation',
        key='data',
        pattern='*.tif',
        auto_mask=None,
        number_of_spatial_samples=None,
        maximum_number_of_iterations=None,
        number_of_resolutions=None,
        initialize_offsets_method=None,
        initialize_offsets_kwargs=None,
        gaussian_sigma=0.,
        use_clahe=False,
        use_edges=False,
        z_range=None,
        z_step=1,
        apply_z_step=False,
        average_for_z_step=False,
        determine_bounds=False,
        parameter_map=None,
        quiet=False,
        overwrite=False,
        n_workers=os.cpu_count(),
        verbose=False,
        debug=False
):

    if not overwrite and os.path.exists(out_filepath):
        print(f'Target file exists: {out_filepath}\nSkipping elastix stack alignment workflow ...')
        return

    from squirrel.library.io import load_data_handle
    from squirrel.library.affine_matrices import AffineStack
    from squirrel.library.data import norm_z_range

    stack, stack_size = load_data_handle(stack, key=key, pattern=pattern)

    transforms = AffineStack(is_sequenced=False, pivot=[0., 0.])
    bounds = []

    z_range = norm_z_range(z_range, stack_size[0])

    if n_workers == 1:
        for idx in range(*z_range, z_step):
            z_slice_fixed = None
            if average_for_z_step:
                z_slice_moving = np.mean(stack[idx: idx + z_step], axis=0).astype('uint8')
                if idx > 0:
                    z_slice_fixed = np.mean(stack[idx - z_step: idx], axis=0).astype('uint8')
            else:
                z_slice_moving = stack[idx]
                if idx > 0:
                    z_slice_fixed = stack[idx - z_step]

            result_matrix, this_bounds = _elastix_one_slice(
                idx,
                z_slice_moving,
                z_slice_fixed,
                z_range,
                determine_bounds,
                transform,
                auto_mask,
                number_of_spatial_samples,
                maximum_number_of_iterations,
                number_of_resolutions,
                initialize_offsets_method,
                initialize_offsets_kwargs,
                parameter_map,
                gaussian_sigma,
                use_clahe,
                use_edges,
                verbose=verbose,
                debug=debug, out_filepath=out_filepath, quiet=quiet
            )

            transforms.append(result_matrix)
            if determine_bounds:
                bounds.append(this_bounds)

    else:
        print(f'Elastix alignment with {n_workers} workers...')
        from multiprocessing import Pool
        with Pool(processes=n_workers) as p:
            tasks = []
            for idx in range(*z_range, z_step):
                z_slice_fixed = None
                if average_for_z_step:
                    z_slice_moving = np.mean(stack[idx: idx + z_step], axis=0).astype('uint8')
                    if idx > 0:
                        z_slice_fixed = np.mean(stack[idx - z_step: idx], axis=0).astype('uint8')
                else:
                    z_slice_moving = stack[idx]
                    if idx > 0:
                        z_slice_fixed = stack[idx - z_step]
                tasks.append(
                    p.apply_async(
                        _elastix_one_slice, (
                            idx,
                            z_slice_moving,
                            z_slice_fixed,
                            z_range,
                            determine_bounds,
                            transform,
                            auto_mask,
                            number_of_spatial_samples,
                            maximum_number_of_iterations,
                            number_of_resolutions,
                            initialize_offsets_method,
                            initialize_offsets_kwargs,
                            parameter_map,
                            gaussian_sigma,
                            use_clahe,
                            use_edges,
                        ), dict(verbose=verbose, debug=debug, out_filepath=out_filepath, quiet=quiet)
                    )
                )
            results = [task.get() for task in tasks]

        for result_matrix, this_bounds in results:
            transforms.append(result_matrix)
            if determine_bounds:
                bounds.append(this_bounds)

    if z_step > 1:
        transforms.set_meta('z_step', z_step)
        if apply_z_step:
            transforms = transforms.get_sequenced_stack()
            transforms = transforms.apply_z_step()

    if determine_bounds:
        assert len(transforms) == len(bounds)
        transforms.set_meta('bounds', np.array(bounds))
    transforms.to_file(out_filepath)


# TODO: write out displacements/errors
def stack_alignment_validation_workflow(
        stack,
        out_dirpath,
        rois,
        key='data',
        pattern='*.tif',
        resolution_yx=(1.0, 1.0),
        out_name=None,
        y_max=None,
        method='elastix',
        gaussian_sigma=1.0,
        subtract_average=False,
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
    from vigra.filters import gaussianSmoothing

    if not os.path.exists(out_dirpath):
        os.mkdir(out_dirpath)

    if out_name is None:
        transforms_dirpath = os.path.join(out_dirpath, 'transforms')
        plot_filepath = os.path.join(out_dirpath, 'plot.pdf')
        image_dirpath = os.path.join(out_dirpath, 'images')
        errors_filepath = os.path.join(out_dirpath, 'errors.csv')
    else:
        transforms_dirpath = os.path.join(out_dirpath, f'transforms-{out_name}')
        plot_filepath = os.path.join(out_dirpath, f'plot-{out_name}.pdf')
        image_dirpath = os.path.join(out_dirpath, f'images-{out_name}')
        errors_filepath = os.path.join(out_dirpath, f'errors-{out_name}.csv')
    if not os.path.exists(image_dirpath):
        os.mkdir(image_dirpath)
    if not os.path.exists(transforms_dirpath):
        os.mkdir(transforms_dirpath)
    image_filepath = os.path.join(image_dirpath, 'image_{:04d}.h5')
    input_filepath = os.path.join(image_dirpath, 'input_{:04d}.h5')
    transforms_filepath = os.path.join(transforms_dirpath, 'transforms_{:04d}.json')

    stack, stack_size = load_data_handle(stack, key=key, pattern=pattern)
    labels = []

    # phase_cross_correlation = None
    xcorr = None
    register_with_sift = None
    if method == 'xcorr':
        from squirrel.library.xcorr import xcorr
    #     from skimage.registration import phase_cross_correlation
    if method == 'sift':
        from squirrel.library.sift2d import register_with_sift2 as register_with_sift

    all_errors = dict()

    for roi_idx, roi in enumerate(rois):

        this_transforms_fp = transforms_filepath.format(roi_idx)

        if not os.path.exists(this_transforms_fp):

            print(f'roi_idx = {roi_idx} / {len(rois) - 1}')

            if verbose:
                print(f'roi = {roi}')
            roi_data = stack[roi]
            transforms = AffineStack(is_sequenced=False, pivot=[0., 0.])
            transforms.append(AffineMatrix(parameters=[1., 0., 0., 0., 1., 0.]))

            for idx in range(len(roi_data) - 1):

                if verbose:
                    print(f'idx = {idx} / {len(roi_data) - 2}')

                z_slice_fixed = roi_data[idx]
                z_slice_moving = roi_data[idx + 1]

                if method == 'elastix':

                    result_matrix, _ = register_with_elastix(
                        z_slice_fixed,
                        z_slice_moving,
                        transform='translation',
                        automatic_transform_initialization=False,
                        auto_mask=False,
                        # number_of_spatial_samples=256,
                        # maximum_number_of_iterations=256,
                        # number_of_resolutions=1,
                        return_result_image=True,
                        params_to_origin=True,
                        gaussian_sigma=2.0,
                        verbose=False  # This produces a ton of output and I don't think I need it here
                    )

                elif method == 'xcorr':
                    shift, error, diffphase = xcorr(
                        z_slice_fixed, z_slice_moving, sigma=gaussian_sigma
                    )
                    # shift, error, diffphase = phase_cross_correlation(
                    #     z_slice_fixed, z_slice_moving,
                    #     upsample_factor=10
                    # )
                    print(f'shift = {shift}')
                    print(f'diffphase = {diffphase}')
                    result_matrix = -AffineMatrix(parameters=[1, 0, shift[0], 0, 1, shift[1]])

                elif method == 'sift':
                    result_matrix = AffineMatrix(parameters=register_with_sift(
                        z_slice_fixed, z_slice_moving  # , transform='translation'
                    ).flatten())

                if verbose:
                    print(f'result_matix = {result_matrix}')
                transforms.append(result_matrix)
                if verbose:
                    print(f'len(transforms) = {len(transforms)}')

                # result_volume.append(result_image)

            if subtract_average:
                transforms = transforms * -transforms.get_smoothed_stack(8)

            result_volume = apply_stack_alignment(
                roi_data,
                roi_data.shape,
                transforms,
                n_workers=1,
                verbose=verbose
            )
            with File(image_filepath.format(roi_idx), mode='w') as f:
                f.create_dataset('data', data=result_volume, compression='gzip')
            with File(input_filepath.format(roi_idx), mode='w') as f:
                f.create_dataset('data', data=roi_data, compression='gzip')
            transforms.to_file(this_transforms_fp)

        else:
            transforms = AffineStack(filepath=this_transforms_fp)

        translations = np.array(transforms.get_translations()) * resolution_yx
        errors = np.sqrt(translations[:, 0] ** 2 + translations[:, 1] ** 2)
        labels.append('roi-{}-mean={:.2f}-median={:.2f}'.format(roi_idx, np.mean(errors), np.median(errors)))
        plt.plot(errors, label=labels[-1])

        all_errors[f'roi_{roi_idx}'] = dict(
            translations=translations.tolist(),
            errors=errors.tolist()
        )

    import json
    with open(errors_filepath, 'w') as f:
        json.dump(all_errors, f, indent=2)

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
                target_image_shape = image_shape
                print(f'found image_shape = {image_shape}')
            else:
                image_shape = target_image_shape
            if not stack.is_sequenced:
                stack = stack.get_sequenced_stack()
            stacks.append(ElastixStack(stack=stack, image_shape=image_shape))

    if verbose:
        print(f'target_image_shape = {target_image_shape}')

    emss = ElastixMultiStepStack(stacks=stacks, image_shape=target_image_shape)

    result_volume = emss.apply_on_image_stack(
        image_stack_h if n_workers == 1 else image_stack,
        target_image_shape=target_image_shape,
        z_range=z_range,
        key=key,
        pattern=pattern,
        n_workers=n_workers,
        quiet=quiet,
        verbose=verbose
    )

    if write_result:
        # from squirrel.library.io import write_h5_container
        from squirrel.library.io import write_stack
        # write_h5_container(out_filepath, result_volume)
        write_stack(out_filepath, result_volume)
        return result_volume
    else:
        return result_volume


def make_elastix_default_parameter_file_workflow(
        out_filepath,
        transform='translation',
        elastix_parameters=None,
        verbose=False
):

    def set_elastix_parameters_from_input(elx_inputs, elx_params):

        if elx_inputs is None:
            return elx_params

        for elx_input in elx_inputs:
            key, values = str.split(elx_input, ':')
            values = str.split(values, ',')
            elx_params[key] = values
            if verbose:
                print(f'{key} = {values}')

        return elx_params

    if transform.startswith('amst-'):
        from squirrel.workflows.amst import get_default_parameters
        params = get_default_parameters(transform.split(sep='-')[1])
    else:
        from SimpleITK import GetDefaultParameterMap
        params = GetDefaultParameterMap(transform)
    set_elastix_parameters_from_input(elastix_parameters, params)
    from SimpleITK import WriteParameterFile
    if verbose:
        print(f'params = {params}')
        print(f'out_filepath = {out_filepath}')
    WriteParameterFile(params, out_filepath)


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
        '/media/julian/Data/projects/hennies/amst_devel/amst2-test-auto-init/tiffs',
        ['/media/julian/Data/projects/hennies/amst_devel/amst2-test-auto-init/result.json'],
        '/media/julian/Data/projects/hennies/amst_devel/amst2-test-auto-init/result',
        auto_pad=False,
        target_image_shape=None,
        z_range=None,
        n_workers=16,
        quiet=False,
        verbose=False,
    )

    # elastix_stack_alignment_workflow(
    #     '/media/julian/Data/projects/woller/problem-area2/subsample',
    #     '/media/julian/Data/projects/woller/problem-area2/test-align-elastix-sbs/tmp-transforms.json',
    #     auto_mask='non-zero',
    #     maximum_number_of_iterations=256,
    #     number_of_resolutions=2,
    #     initialize_offsets_method='init_elx',
    #     gaussian_sigma=2,
    #     determine_bounds=True,
    #     n_workers=1
    # )

    # register_with_elastix_workflow(
    #     '/media/julian/Data/projects/schneider/DAPI_R17-2-bin_xy4_z2.tif',
    #     '/media/julian/Data/projects/schneider/Dapi_R1-2-bin_xy4_z2.tif',
    #     '/media/julian/Data/projects/schneider/transform-init_elx4.json',
    #     out_img_filepath='/media/julian/Data/projects/schneider/registered-inti_elx4.tif',
    #     transform='translation',
    #     auto_mask='non-zero',
    #     number_of_spatial_samples=None,
    #     maximum_number_of_iterations=None,
    #     number_of_resolutions=2,
    #     initialize_offsets_method='init_elx',
    #     initialize_offsets_kwargs=dict(
    #         spacing=16,
    #         binning=8,
    #         elx_binning=1,
    #         mi_thresh=-3,
    #     ),
    #     gaussian_sigma=2.,
    #     use_clahe=False,
    #     use_edges=False,
    #     parameter_map=None,
    #     n_workers=os.cpu_count(),
    #     debug_dirpath='/media/julian/Data/projects/schneider/debug',
    #     verbose=False
    # )
