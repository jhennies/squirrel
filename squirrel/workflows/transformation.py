
import numpy as np


def decompose_affine(
        transform,
        out_folder=None,
        shear_to_translation_pivot=None,
        pivot=None,
        verbose=False
):
    from ..library.transformation import (
        load_transform_matrix,
        extract_approximate_rotation_affine,
        decompose_3d_transform,
        validate_and_reshape_matrix
    )
    from ..library.elastix import save_transforms

    if type(transform) == str:
        transform = load_transform_matrix(transform)

    if verbose:
        print(f'transform = {transform}')

    transform = validate_and_reshape_matrix(transform, 3)

    if verbose:
        print(f'transform = {transform}')

    if shear_to_translation_pivot is not None:
        tpivot = np.array(
            [
                [1, 0, 0, shear_to_translation_pivot[0]],
                [0, 1, 0, shear_to_translation_pivot[1]],
                [0, 0, 1, shear_to_translation_pivot[2]],
                [0, 0, 0, 1]
            ]
        )
        tpivot_ = np.array(
            [
                [1, 0, 0, -shear_to_translation_pivot[0]],
                [0, 1, 0, -shear_to_translation_pivot[1]],
                [0, 0, 1, -shear_to_translation_pivot[2]],
                [0, 0, 0, 1]
            ]
        )
        transform = np.dot(transform, tpivot)
    if verbose:
        print(f'transform = {transform}')
    decomposition = decompose_3d_transform(transform, verbose=verbose)
    if verbose:
        print(f'decomposition = {decomposition}')

    # if shear_to_translation_pivot is not None:
    #     import math
    #     shear = decomposition[3]
    #     translation = decomposition[0]
    #     translation[0] += shear_to_translation_pivot[0] * math.atan(shear[0])
    #     translation[1] += shear_to_translation_pivot[1] * math.atan(shear[1])
    #     translation[2] += shear_to_translation_pivot[2] * math.atan(shear[2])

    from ..library.transformation import (
        setup_translation_matrix,
        setup_scale_matrix,
        setup_shear_matrix
    )

    translation = decomposition[0]
    translation = setup_translation_matrix(translation)
    if shear_to_translation_pivot is not None:
        translation = np.dot(translation, tpivot_)
    scale = decomposition[2]
    scale = setup_scale_matrix(scale)
    rotation = np.concatenate([decomposition[1], np.swapaxes([[0., 0., 0.]], 0, 1)], axis=1)
    shear = decomposition[3]
    shear = setup_shear_matrix(shear)

    if verbose:
        print(f'translation = {translation}')
        print(f'rotation = {rotation}')
        print(f'scale = {scale}')
        print(f'shear = {shear}')

    if out_folder is not None:
        import os
        save_transforms(translation, os.path.join(out_folder, 'translation.json'), param_order='M', save_order='C', ndim=3, verbose=verbose)
        save_transforms(rotation, os.path.join(out_folder, 'rotation.json'), param_order='M', save_order='C', ndim=3, verbose=verbose)
        save_transforms(scale, os.path.join(out_folder, 'scale.json'), param_order='M', save_order='C', ndim=3, verbose=verbose)
        save_transforms(shear, os.path.join(out_folder, 'shear.json'), param_order='M', save_order='C', ndim=3, verbose=verbose)
    return decomposition


def apply_affine(
        image,
        transform,
        out_filepath=None,
        image_key='data',
        no_offset_to_center=False,
        pivot=None,
        apply='all',  # Can be ['all' | 'rotation']
        scale_canvas=False,
        verbose=False
):

    if verbose:
        print(f'image = {image if type(image) == str else image.shape}')
        print(f'transform = {transform}')
        print(f'out_filepath = {out_filepath}')
        print(f'image_key = {image_key}')

    from ..library.transformation import apply_affine_transform
    from ..library.io import load_data, write_h5_container

    if type(image) == str:
        image = load_data(image, key=image_key)

    if type(transform) == str:
        from ..library.transformation import load_transform_matrix
        transform = load_transform_matrix(transform)

    result = apply_affine_transform(
        image,
        transform,
        no_offset_to_center=no_offset_to_center,
        pivot=pivot,
        apply=apply,
        scale_canvas=scale_canvas,
        verbose=verbose
    )

    write_h5_container(out_filepath, result, key='data')

    return result


def apply_sequential_affine(
        image,
        transforms,
        out_filepath=None,
        image_key='data',
        no_offset_to_center=False,
        pivot=None,
        verbose=False
):

    from ..library.transformation import load_transform_matrix, validate_and_reshape_matrix
    if len(transforms) == 1:
        transforms = load_transform_matrix(transforms[0])
        transform = validate_and_reshape_matrix(transforms[0], 3)
        for t in transforms[1:]:
            transform = np.dot(transform, validate_and_reshape_matrix(t, 3))

    else:
        transform = validate_and_reshape_matrix(load_transform_matrix(transforms[0]), 3)
        for t in transforms[1:]:
            t = validate_and_reshape_matrix(load_transform_matrix(t), 3)
            transform = np.dot(transform, t)

    from ..library.elastix import save_transforms
    import os
    save_transforms(
        transform[:3, :],
        os.path.join(
            os.path.split(out_filepath)[0],
            os.path.splitext(os.path.split(out_filepath)[1])[0] + '.json'
        ),
        param_order='M',
        save_order='C',
        ndim=3,
        verbose=verbose
    )

    apply_affine(
        image,
        transform,
        out_filepath=out_filepath,
        image_key=image_key,
        no_offset_to_center=no_offset_to_center,
        pivot=pivot,
        verbose=verbose
    )


def apply_average_affine(
        image,
        transform,
        out_filepath=None,
        image_key='data',
        method='mean',
        verbose=False
):

    if type(transform) == str:
        import json
        with open(transform, mode='r') as f:
            transform = np.array(json.load(f))

    if method == 'mean':
        transform = np.mean(transform, axis=0)
    elif method == 'median':
        transform = np.median(transform, axis=0)
    else:
        raise ValueError(f'Invalid averaging method: {method}')

    return apply_affine(
        image,
        transform,
        out_filepath=out_filepath,
        image_key=image_key,
        verbose=verbose
    )


def apply_from_z_chunks(
        image,
        transforms,
        chunk_distance,
        out_filepath=None,
        image_key='data',
        apply=('rotate', 'scale', 'translate', 'shear'),  # also: ('all', 'scale-xy/-z')
        pivot=(0., 0., 0.),
        verbose=False
):

    from ..library.transformation import (
        decompose_3d_transform,
        validate_and_reshape_matrix,
        setup_translation_matrix,
        setup_rotation_matrix,
        setup_scale_matrix,
        setup_shear_matrix
    )
    from ..library.elastix import get_affine_rotation_parameters, save_transforms

    if type(transforms) == str:
        import json
        with open(transforms, mode='r') as f:
            transforms = json.load(f)

    def _matrices_from_affine(transforms):

        translations = []
        rotations = []
        scales = []
        shears = []

        for idx, t in enumerate(transforms):
            t = validate_and_reshape_matrix(t, 3)
            translation, rotation, scale, shear = decompose_3d_transform(t)
            translations.append(translation)
            rotations.append(rotation)
            scales.append(scale)
            shears.append(shear)

            print(f'translation = {translation}')
            print(f'rotation = {rotation}')
            print(f'scale = {scale}')
            print(f'shear = {shear}')

        # Rotation and translation are just averaged from each chunk
        rotation = setup_rotation_matrix(np.mean(rotations, axis=0))
        translation = setup_translation_matrix(np.mean(translations, axis=0))

        # z-scale is handled individually from the z-translation of first and last chunk
        stack_height = chunk_distance * len(translations)
        z_scale = stack_height / (stack_height + translations[0][0] - translations[-1][0])
        z_scale = setup_scale_matrix([z_scale, z_scale, z_scale])
        if verbose:
            print(f'z_scale = {z_scale}')

        # x- and y- scales are averaged
        xy_scale = setup_scale_matrix(np.mean(scales, axis=0))
        xy_scale[0, 0] = 0.

        # The xy-shearing is defined by the x- and y- translations
        # TODO! but ignoring for now

        return rotation, translation, z_scale, xy_scale

    def _matrices_from_similarity(transforms):

        translations = []
        rotations = []
        scales = []

        for idx, t in enumerate(transforms):
            rotations.append(t[:3])
            translations.append(t[3:6][::-1])
            scales.append(t[6])

        # Rotation and translation are just averaged from each chunk
        rotation = np.mean(rotations, axis=0)
        translation = np.mean(translations, axis=0)

        # z-scale is handled individually from the z-translation of first and last chunk
        stack_height = chunk_distance * len(translations)
        z_scale = stack_height / (stack_height + translations[0][0] - translations[-1][0])

        # x- and y- scales are averaged
        xy_scale = np.mean(scales, axis=0)

        # Set up the matrices
        rotation = save_transforms(
            list(get_affine_rotation_parameters(rotation)) + [0., 0., 0.],
            None, param_order='elastix', save_order='M', ndim=3, verbose=verbose
        )
        translation = setup_translation_matrix(translation)
        xy_scale = setup_scale_matrix([1., xy_scale, xy_scale])
        z_scale = setup_scale_matrix([z_scale, z_scale, z_scale])

        return rotation, translation, z_scale, xy_scale

    if len(transforms[0]) == 12:
        rotation, translation, z_scale, xy_scale = _matrices_from_affine(transforms)
    elif len(transforms[0]) == 7:
        rotation, translation, z_scale, xy_scale = _matrices_from_similarity(transforms)
    else:
        raise ValueError(f'transform has invalid length: {len(transforms[0])}')

    if verbose:
        print(f'rotation = {rotation}')
        print(f'translation = {translation}')
        print(f'z_scale = {z_scale}')
        print(f'xy_scale = {xy_scale}')

    rotation = validate_and_reshape_matrix(rotation, 3)
    translation = validate_and_reshape_matrix(translation, 3)
    z_scale = validate_and_reshape_matrix(z_scale, 3)
    xy_scale = validate_and_reshape_matrix(xy_scale, 3)

    if verbose:
        print(f'rotation = {rotation}')
        print(f'translation = {translation}')
        print(f'z_scale = {z_scale}')
        print(f'xy_scale = {xy_scale}')

    transform = rotation if 'rotate' in apply else np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    assert transform[0, 3] == 0.
    assert transform[1, 3] == 0.
    assert transform[2, 3] == 0.

    if 'translate' in apply:
        if verbose:
            print(f'transform = {transform}')
            print(f'translation = {translation}')
        transform = np.dot(transform, translation)
    if 'scale-z' in apply or 'scale' in apply:
        transform = np.dot(transform, z_scale)
    if 'scale-xy' in apply or 'scale' in apply:
        transform = np.dot(transform, xy_scale)

    from ..library.elastix import save_transforms
    import os
    save_transforms(
        transform[:3, :],
        os.path.join(
            os.path.split(out_filepath)[0],
            os.path.splitext(os.path.split(out_filepath)[1])[0] + '.json'
        ),
        param_order='M', save_order='C', ndim=3, verbose=verbose
    )

    return apply_affine(
        image,
        transform,
        out_filepath=out_filepath,
        image_key=image_key,
        no_offset_to_center=False,
        pivot=pivot,
        verbose=verbose
    )


def apply_rotation_and_scale_from_transform_stack(
        image,
        transform_stack,
        stack_item_distance,  # The distance in pixels
        out_filepath=None,
        image_key='data',
        apply=('rotate', 'scale', 'translate'),
        pivot=(0., 0., 0.),
        verbose=False
):
    """
    Note that apply=('scale') applies the z-scale to all dimensions.
    """

    if verbose:
        print(f'apply = {apply}')

    from ..library.transformation import extract_approximate_rotation_affine
    from ..library.transformation import validate_and_reshape_matrix

    if type(transform_stack) == str:
        import json
        with open(transform_stack, mode='r') as f:
            transform_stack = json.load(f)

    if verbose:
        print(f'transform_stack = {transform_stack}')
    rotation = []
    z_displacements = []
    y_displacements = []
    x_displacements = []

    for idx, t in enumerate(transform_stack):
        t = validate_and_reshape_matrix(t, 3)
        rotation.append(extract_approximate_rotation_affine(t, 0))
        z_displacements.append(t[0, 3])
        y_displacements.append(t[1, 3])
        x_displacements.append(t[2, 3])
        if verbose:
            print(f't = {t}')
    rotation = np.mean(rotation, axis=0)
    if verbose:
        print(f'rotation = {rotation}')
        print(f'x_displacements = {x_displacements}')
        print(f'y_displacements = {y_displacements}')
        print(f'z_displacements = {z_displacements}')

    stack_height = stack_item_distance * len(z_displacements)
    z_scale = stack_height / (stack_height + z_displacements[0] - z_displacements[-1])
    if verbose:
        print(f'z_scale = {z_scale}')

    z_displacement = np.mean(z_displacements)
    y_displacement = np.mean(y_displacements)
    x_displacement = np.mean(x_displacements)

    transform = rotation if 'rotate' in apply else np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
    assert transform[0, 3] == 0.
    assert transform[1, 3] == 0.
    assert transform[2, 3] == 0.
    if 'scale-z' in apply or 'scale' in apply:
        print('scale-z')
        transform[0, 0] = z_scale
        transform[0, 3] += pivot[0] - pivot[0] * z_scale
    if 'scale-y' in apply or 'scale' in apply:
        print('scale-y')
        transform[1, 1] = z_scale
        transform[1, 3] += pivot[1] - pivot[1] * z_scale
    if 'scale-x' in apply or 'scale' in apply:
        print('scale-x')
        transform[2, 2] = z_scale
        transform[2, 3] += pivot[2] - pivot[2] * z_scale
    if 'translate-z' in apply or 'translate' in apply:
        print('translate-z')
        transform[0, 3] += z_displacement
    if 'translate-y' in apply or 'translate' in apply:
        print('translate-y')
        transform[1, 3] += y_displacement
    if 'translate-x' in apply or 'translate' in apply:
        print('translate-x')
        transform[2, 3] += x_displacement

    from ..library.elastix import save_transforms
    import os
    save_transforms(
        transform[:3, :], os.path.join(os.path.split(out_filepath)[0], 'transform.json'),
        param_order='M', save_order='C', ndim=3, verbose=verbose
    )

    return apply_affine(
        image,
        transform,
        out_filepath=out_filepath,
        image_key=image_key,
        no_offset_to_center=True,
        pivot=None,
        verbose=verbose
    )

