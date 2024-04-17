
import SimpleITK
import numpy as np
import os


def _load_data(filepath, key='data'):

    from ..library.io import get_filetype

    if get_filetype(filepath) == 'h5':
        from ..library.io import load_h5_container
        return load_h5_container(filepath, key)
    if get_filetype(filepath) == 'nii':
        from ..library.io import load_nii_file
        return load_nii_file(filepath)


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


def slices_to_volume(
        moving_filepath,
        fixed_filepath,
        out_filepath,
        moving_key='data',
        fixed_key='data',
        z_chunk_size=16,
        transform='affine',
        automatic_transform_initialization=False,
        verbose=False
):

    from ..library.elastix import register_with_elastix, save_transforms
    from ..library.io import write_h5_container
    from ..library.io import make_directory

    out_basename = os.path.splitext(out_filepath)[0]

    elastix_cache = out_basename + '.elastix_cache'
    make_directory(elastix_cache, exist_ok=True)

    fixed_volume = _load_data(fixed_filepath, key=fixed_key)
    moving_volume = _load_data(moving_filepath, key=moving_key)

    def _cast_to_uint8(image):
        if image.dtype != 'uint8':
            print(f'Warning: Input image was not uint8. Casting the filetype, however this may cause unintended effects')
            return image.astype('uint8')
        return image

    fixed_volume = _cast_to_uint8(fixed_volume)
    moving_volume = _cast_to_uint8(moving_volume)

    assert fixed_volume.shape[0] >= moving_volume.shape[0]

    result_volume = []
    result_transforms = []

    for zidx, z_slice_moving in enumerate(moving_volume):

        z_slice_fixed = fixed_volume[zidx]

        result_dict = register_with_elastix(
            z_slice_fixed, z_slice_moving,
            transform=transform,
            automatic_transform_initialization=automatic_transform_initialization,
            out_dir=elastix_cache,
            params_to_origin=True,
            verbose=verbose
        )

        result_volume.append(result_dict['result_image'])
        result_transforms.append(
            save_transforms(
                result_dict['affine_parameters'], None,
                param_order='M', save_order='C', ndim=2, verbose=verbose
            ).tolist()
        )
        # result_transforms.append(result_dict['affine_parameters'][:2])

    import json
    with open(out_basename + '.json', mode='w') as f:
        json.dump(result_transforms, f, indent=2)

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
        z_range=None,
        determine_bounds=False,
        verbose=False
):

    if os.path.exists(out_filepath):
        print(f'Target file exists: {out_filepath}\nSkipping elastix stack alignment workflow ...')
        return

    from ..library.io import load_data_handle, load_data_from_handle_stack
    from ..library.elastix import register_with_elastix
    from ..library.affine_matrices import AffineMatrix, AffineStack
    from ..library.data import norm_z_range

    stack, stack_size = load_data_handle(stack, key=key, pattern=pattern)

    transforms = AffineStack(is_sequenced=False, pivot=[0., 0.])
    bounds = []

    z_range = norm_z_range(z_range, stack_size[0])

    for idx in range(*z_range):

        print(f'idx = {idx} / {z_range[1]}')
        z_slice_moving, _ = load_data_from_handle_stack(stack, idx)

        if idx == 0:
            transforms.append(AffineMatrix([1., 0., 0., 0., 1., 0.], pivot=[0., 0.]))
        else:

            z_slice_fixed, _ = load_data_from_handle_stack(stack, idx - 1)

            result_matrix, _ = register_with_elastix(
                z_slice_fixed,
                z_slice_moving,
                transform=transform,
                automatic_transform_initialization=False,
                # params_to_origin=True,
                auto_mask=auto_mask,
                number_of_spatial_samples=number_of_spatial_samples,
                maximum_number_of_iterations=maximum_number_of_iterations,
                number_of_resolutions=number_of_resolutions,
                pre_fix_big_jumps=pre_fix_big_jumps,
                return_result_image=False,
                verbose=verbose
            )
            result_matrix.shift_pivot_to_origin()
            transforms.append(result_matrix)

        if determine_bounds:
            from ..library.image import get_bounds
            bounds.append(get_bounds(z_slice_moving, return_ints=True))

    transforms.set_meta('bounds', np.array(bounds))
    transforms.to_file(out_filepath)

