
import numpy as np


def apply_affine(
        image,
        transform,
        out_filepath=None,
        image_key='data',
        no_offset_to_center=False,
        pivot=None,
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

