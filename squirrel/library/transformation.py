import math

import numpy as np


def transform_matrix_offset_center(matrix, shape):

    o_x = float(shape[0]) / 2  # + 0.5
    o_y = float(shape[1]) / 2  # + 0.5
    o_z = float(shape[2]) / 2
    offset_matrix = np.array([[1, 0, 0, o_x],
                              [0, 1, 0, o_y],
                              [0, 0, 1, o_z],
                              [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x],
                             [0, 1, 0, -o_y],
                             [0, 0, 1, -o_z],
                             [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def validate_matrix(matrix, ndim):
    matrix = np.array(matrix)
    if matrix.ndim == 1:
        assert len(matrix) == ndim ** 2 + ndim or len(matrix) == (ndim + 1) ** 2, \
            f'Length of matrix={len(matrix)} does not match image dimension={ndim}!'
    if matrix.ndim == 2:
        assert matrix.shape == (ndim, ndim + 1) or matrix.shape == (ndim + 1, ndim + 1), \
            f'Matrix shape={matrix.shape} does not match image dimension={ndim}'


def load_transform_matrix(filepath):

    from squirrel.library.io import get_filetype
    filetype = get_filetype(filepath)

    if filetype == 'json':
        import json
        with open(filepath, mode='r') as f:
            return json.load(f)

    if filetype == 'csv':
        from numpy import genfromtxt
        return genfromtxt(filepath, delimiter=',')


def validate_and_reshape_matrix(matrix, ndim):

    matrix = np.array(matrix)
    validate_matrix(matrix, ndim)

    if matrix.ndim == 1:
        try:
            matrix = np.reshape(matrix, (ndim, ndim + 1))
            return np.concatenate([matrix, [[0] * ndim + [1]]], axis=0)
        except ValueError:
            return np.reshape(matrix, (ndim + 1, ndim + 1))

    if matrix.ndim == 2:
        if matrix.shape == (ndim + 1, ndim + 1):
            return matrix
        if matrix.shape == (ndim, ndim + 1):
            return np.concatenate([matrix, [[0] * ndim + [1]]], axis=0)
        raise ValueError(f'Matrix shape={matrix.shape} does not match image dimension={ndim}')

    raise ValueError(f'Matrix has invalid number of dimensions: {matrix.ndim}')


def setup_translation_matrix(translation_zyx):

    return np.array(
        [
            [1., 0., 0., translation_zyx[0]],
            [0., 1., 0., translation_zyx[1]],
            [0., 0., 1., translation_zyx[2]]
        ]
    )


def setup_rotation_matrix(rotation):
    return np.concatenate((rotation, np.swapaxes([[0., 0., 0.]], 0, 1)), axis=1)


def setup_scale_matrix(scale_zyx):

    return np.array(
        [
            [scale_zyx[0], 0., 0., 0.],
            [0., scale_zyx[1], 0., 0.],
            [0., 0., scale_zyx[2], 0.]
        ]
    )


def setup_shear_matrix(shear_zyx):

    return np.array(
        [
            [1., shear_zyx[0], shear_zyx[1], 0.],
            [0., 1., shear_zyx[2], 0.],
            [0., 0., 1., 0.]
        ]
    )


def decompose_3d_transform(transform, return_matrices=False, verbose=False):

    from transforms3d.affines import decompose

    transform = validate_and_reshape_matrix(transform, ndim=3)
    decomp = decompose(transform)

    if not return_matrices:
        return decomp

    return (
        setup_translation_matrix(decomp[0]),
        decomp[1],
        setup_scale_matrix(decomp[2]),
        setup_shear_matrix(decomp[3])
    )


def extract_approximate_rotation_affine(transform, coerce_affine_dimension):
    print(transform)

    from copy import deepcopy
    new_transform = deepcopy(transform)

    # for c in range(3):
    #     sq_sum = 0
    #     for r in range(3):
    #         sq_sum += new_transform[r, c] ** 2
    #     s = 1. / math.sqrt(sq_sum)
    #     for r in range(3):
    #         new_transform[r, c] *= s

    x = new_transform[:3, 0]
    y = new_transform[:3, 1]
    z = new_transform[:3, 2]

    if coerce_affine_dimension == 0:
        x = np.cross(y, z)
        z = np.cross(x, y)
    if coerce_affine_dimension == 1:
        y = np.cross(z, x)
        x = np.cross(y, z)
    if coerce_affine_dimension == 2:
        z = np.cross(x, y)
        y = np.cross(z, x)

    new_transform[:3, 0] = x
    new_transform[:3, 1] = y
    new_transform[:3, 2] = z

    new_transform[0, 0] = 1.0
    new_transform[1, 1] = 1.0
    new_transform[2, 2] = 1.0

    new_transform[:3, 3] = 0.0

    return new_transform


def apply_affine_transform(
        x,
        transform_matrix,
        fill_mode='nearest',
        cval=0.,
        order=1,
        no_offset_to_center=False,
        pivot=None,
        apply='all',
        scale_canvas=False,
        verbose=False
):

    transform_matrix_ = validate_and_reshape_matrix(transform_matrix, x.ndim)

    if verbose:
        print(f'transform_matrix = {transform_matrix_}')
        print(f'x.shape = {x.shape}')
    if pivot is None and not no_offset_to_center:
        transform_matrix_ = transform_matrix_offset_center(transform_matrix_, x.shape)
    if pivot is not None:
        transform_matrix_ = transform_matrix_offset_center(transform_matrix_, np.array(pivot) * 2)

    if apply == 'rotation':
        transform_matrix_ = extract_approximate_rotation_affine(transform_matrix_, 0)

    import scipy.ndimage as ndi
    x = ndi.affine_transform(
        x,
        transform_matrix_,
        order=order,
        mode=fill_mode,
        cval=cval)

    if scale_canvas:
        assert no_offset_to_center and pivot is None, 'Canvas scaling only implemented when scaling with reference to image origin!'
        _, _, scale, _ = decompose_3d_transform(transform_matrix_)
        if verbose:
            print(f'scale = {scale}')
        crop = (np.ceil(np.array(x.shape) / np.array(scale))).astype(int)
        if verbose:
            print(f'crop = {crop}')
        x = x[:crop[0], :crop[1], :crop[2]]

    return x
