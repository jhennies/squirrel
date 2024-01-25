
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


def affine3d(
        moving_filepath,
        fixed_filepath,
        out_path,
        moving_key='data',
        fixed_key='data',
        automatic_transform_initialization=False,
        view_results_in_napari=False,
        verbose=False
):

    from ..library.elastix import register_with_elastix

    fixed_image = _load_data(fixed_filepath, key=fixed_key)
    moving_image = _load_data(moving_filepath, key=moving_key)

    result_image, result_transform = register_with_elastix(
        fixed_image, moving_image,
        transform='affine',
        automatic_transform_initialization=automatic_transform_initialization,
        verbose=verbose
    )

    from ..library.io import write_h5_container
    write_h5_container(result_image, result_image, 'data')

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
        automatic_transform_initialization=False,
        view_results_in_napari=False,
        verbose=False
):

    from ..library.elastix import register_with_elastix
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

    z_chunk_size = 16
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

        this_result_image, this_transform = register_with_elastix(
            this_fixed, this_moving,
            out_dir=out_path,
            transform='affine',
            automatic_transform_initialization=automatic_transform_initialization,
            verbose=verbose
        )
        result_images.append(this_result_image)
        result_transforms.append([float(x) for x in this_transform])

        write_h5_container(out_images_filepath, this_result_image, 'result', append=True)

        if verbose:
            print(f'Done with chunk {chunk_start}')

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
