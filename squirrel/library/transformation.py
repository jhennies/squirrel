
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


def apply_affine_transform(
        x,
        transform_matrix,
        fill_mode='nearest',
        cval=0.,
        order=1
):

    transform_matrix_ = validate_and_reshape_matrix(transform_matrix, x.ndim)
    transform_matrix_ = transform_matrix_offset_center(transform_matrix_, x.shape)

    import scipy.ndimage as ndi
    x = ndi.affine_transform(
        x,
        transform_matrix_,
        order=order,
        mode=fill_mode,
        cval=cval)

    return x