import math

import numpy as np


def transform_matrix_offset_center(matrix, shape, ndim=3):

    if ndim == 3:
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

    if ndim == 2:
        o_x = float(shape[0]) / 2  # + 0.5
        o_y = float(shape[1]) / 2  # + 0.5
        offset_matrix = np.array([[1, 0, o_x],
                                  [0, 1, o_y],
                                  [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x],
                                 [0, 1, -o_y],
                                 [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix


# def validate_matrix(matrix, ndim):
#     matrix = np.array(matrix)
#     if matrix.ndim == 1:
#         if len(matrix) == ndim ** 2 + ndim and len(matrix) == (ndim + 1) ** 2:
#             raise ValueError(f'Length of matrix={len(matrix)} does not match image dimension={ndim}!')
#     if matrix.ndim == 2:
#         if matrix.shape != (ndim, ndim + 1) and matrix.shape != (ndim + 1, ndim + 1):
#             raise ValueError(f'Matrix shape={matrix.shape} does not match image dimension={ndim}')
#

# def load_transform_matrix(filepath, validate=False):
#
#     from squirrel.library.io import get_filetype
#     filetype = get_filetype(filepath)
#
#     if filetype == 'json':
#         import json
#         with open(filepath, mode='r') as f:
#             return json.load(f)
#
#     if filetype == 'csv':
#         from numpy import genfromtxt
#         return genfromtxt(filepath, delimiter=',')


# def load_transform_matrices(filepath, validate=False, ndim=3):
#
#     from squirrel.library.io import get_filetype
#     assert get_filetype(filepath) == 'json', 'Multiple transforms in one file only supported for json files'
#
#     import json
#     with open(filepath, mode='r') as f:
#         transforms_info = json.load(f)
#
#     sequenced = False
#     if type(transforms_info) == dict:
#         transforms = transforms_info['transforms']
#         sequenced = transforms_info['sequenced'] if 'sequenced' in transforms_info.keys() else False
#     else:
#         transforms = transforms_info
#
#     if np.array(transforms).ndim == 1:
#         if validate:
#             return [validate_and_reshape_matrix(transforms, ndim)]
#         return transforms
#     if np.array(transforms).ndim >= 2:
#         if not validate:
#             return transforms
#         try:
#             # If this works it was one transform in matrix shape
#             return [validate_and_reshape_matrix(transforms, ndim)]
#         except ValueError:
#             pass
#         out_transforms = []
#         for transform in transforms:
#             out_transforms.append(validate_and_reshape_matrix(transform, ndim))
#         return out_transforms, sequenced
#     raise RuntimeError(f'Invalid file contents for {filepath}')


# def load_transform_matrices_from_multiple_files(filepaths, validate=False, ndim=3):
#
#     all_sequenced = None
#     all_transforms = []
#     for filepath in filepaths:
#         transforms, sequenced = load_transform_matrices(filepath, validate=validate, ndim=ndim)
#         all_sequenced = sequenced if all_sequenced is None else all_sequenced
#         assert sequenced == all_sequenced, 'All of the files loaded into the full sequence must be either sequenced ' \
#                                            'or non-sequenced, not mixed in type!'
#         all_transforms.extend(transforms)
#     return all_transforms, all_sequenced


# def save_transformation_matrices(filepath, transforms, sequenced=None):
#
#     transforms = np.array(transforms)[:, :6].tolist()
#
#     if sequenced is not None:
#         output = dict(
#             transforms=transforms,
#             sequenced=sequenced
#         )
#     else:
#         output = transforms
#
#     import json
#     with open(filepath, mode='w') as f:
#         json.dump(output, f, indent=2)


# def validate_and_reshape_matrix(matrix, ndim):
#
#     matrix = np.array(matrix)
#     validate_matrix(matrix, ndim)
#
#     if matrix.ndim == 1:
#         try:
#             matrix = np.reshape(matrix, (ndim, ndim + 1))
#             return np.concatenate([matrix, [[0] * ndim + [1]]], axis=0)
#         except ValueError:
#             return np.reshape(matrix, (ndim + 1, ndim + 1))
#
#     if matrix.ndim == 2:
#         if matrix.shape == (ndim + 1, ndim + 1):
#             return matrix
#         if matrix.shape == (ndim, ndim + 1):
#             return np.concatenate([matrix, [[0] * ndim + [1]]], axis=0)
#         raise ValueError(f'Matrix shape={matrix.shape} does not match image dimension={ndim}')
#
#     raise ValueError(f'Matrix has invalid number of dimensions: {matrix.ndim}')


def setup_translation_matrix(translation_zyx, ndim=3):

    if ndim == 3:

        return np.array(
            [
                [1., 0., 0., translation_zyx[0]],
                [0., 1., 0., translation_zyx[1]],
                [0., 0., 1., translation_zyx[2]]
            ]
        )

    if ndim == 2:

        return np.array(
            [
                [1., 0., translation_zyx[0]],
                [0., 1., translation_zyx[1]]
            ]
        )

    raise ValueError(f'Invalid number of dimensions = {ndim}')


def setup_rotation_matrix(rotation):
    return np.concatenate((rotation, np.swapaxes([[0., 0., 0.]], 0, 1)), axis=1)


def setup_2d_rotation_matrix_from_angle(angle):
    return np.array([
        [math.cos(angle), math.sin(angle), 0.],
        [-math.sin(angle), math.cos(angle), 0.]
    ])


def setup_scale_matrix(scale_zyx, ndim=3):

    import numbers

    assert len(scale_zyx) == ndim
    assert isinstance(scale_zyx[0], numbers.Number)

    if ndim == 3:
        return np.array(
            [
                [scale_zyx[0], 0., 0., 0.],
                [0., scale_zyx[1], 0., 0.],
                [0., 0., scale_zyx[2], 0.]
            ]
        )

    if ndim == 2:
        return np.array(
            [
                [scale_zyx[0], 0., 0.],
                [0., scale_zyx[1], 0.]
            ]
        )

    raise ValueError(f'Invalid number of dimensions = {ndim}')


def setup_shear_matrix(shear_zyx, ndim=3):

    if ndim == 3:
        return np.array(
            [
                [1., shear_zyx[0], shear_zyx[1], 0.],
                [0., 1., shear_zyx[2], 0.],
                [0., 0., 1., 0.]
            ]
        )

    if ndim == 2:
        return np.array(
            [
                [1., shear_zyx[0], 0.],
                [0., 1., 0.]
            ]
        )
    raise ValueError(f'Invalid number of dimensions = {ndim}')


# def decompose_3d_transform(transform, return_matrices=False, ndim=3, verbose=False):
#
#     from transforms3d.affines import decompose
#
#     transform = validate_and_reshape_matrix(transform, ndim=ndim)
#     decomp = decompose(transform)
#
#     if not return_matrices:
#         return decomp
#
#     return (
#         setup_translation_matrix(decomp[0]),
#         decomp[1],
#         setup_scale_matrix(decomp[2]),
#         setup_shear_matrix(decomp[3])
#     )


# def decompose_sequence(sequence):
#
#     from ..library.elastix import save_transforms
#
#     from transforms3d.affines import decompose
#     translations = []
#     rotations = []
#     zooms = []
#     shears = []
#     for m in sequence:
#         m = save_transforms(m, None, param_order='C', save_order='M', ndim=2)
#         m = validate_and_reshape_matrix(m, 2)
#         t, r, z, s = decompose(m)
#         translations.append(t)
#         rotations.append(r)
#         zooms.append(z)
#         shears.append(s)
#     return translations, rotations, zooms, shears


# def smooth_2d_affine_sequence(
#         sequence,
#         sigma=1.0,
#         components=None
# ):
#
#     from scipy.ndimage import gaussian_filter1d
#     from scipy.ndimage import convolve1d
#
#     def _gaussian_arithmetic(seq):
#         # This on is trivial: the weighted arithmetic mean
#         return gaussian_filter1d(seq, sigma, axis=0)
#
#     # def _gaussian_geometric(seq):
#     #     # Easy too: weighted geometric mean (weights are the normal distribution)
#     #     kernel = [x for x in range(int(np.floor(-sigma * 2)), int(np.ceil(sigma * 2) + 1))]
#     #     kernel = np.array([1/(sigma * math.sqrt(2 * math.pi)) * math.exp(- x ** 2 / (2 * sigma ** 2)) for x in kernel])
#     #     kernel /= kernel.sum()
#     #     return convolve1d(seq, kernel, axis=0)
#     #
#     # def _to_angles(rotations):
#     #     angles00 = [math.acos(x[0, 0]) for x in rotations]
#     #     # angles01 = [math.asin(x[0, 1]) for x in rotations]
#     #     # angles10 = [math.asin(-x[1, 0]) for x in rotations]
#     #     # angles11 = [math.acos(x[1, 1]) for x in rotations]
#     #     for idx, x in enumerate(angles00):
#     #         if -3 * np.pi < x < -np.pi:
#     #             x += 2 * np.pi
#     #         elif -np.pi < x < np.pi:
#     #             pass
#     #         elif np.pi < x < 3 * np.pi:
#     #             x -= 2 * np.pi
#     #         else:
#     #             ValueError(f'Invalid angle: {x}')
#     #         angles00[idx] = x
#     #     for x in angles00:
#     #         assert -np.pi < x < np.pi
#     #     return angles00
#
#     sequence = _gaussian_arithmetic(sequence)
#
#     # if components is None:
#     #     components = ['translation', 'rotation', 'shear', 'scale']
#     #
#     # sequence = np.array(sequence, dtype='float64')
#     # # Decompose matrices
#     # translations, rotations, zooms, shears = decompose_sequence(sequence)
#     # # Convert the rotation matrices to angles
#     # rotations = _to_angles(rotations)
#     # # print(rotations)
#     #
#     # if sigma > 0:
#     #     # Filter the components
#     #     if 'translation' in components:
#     #         translations = _gaussian_arithmetic(translations)
#     #     if 'scale' in components:
#     #         zooms = _gaussian_geometric(zooms)
#     #     if 'rotation' in components:
#     #         rotations = _gaussian_arithmetic(rotations)
#     #     if 'shear' in components:
#     #         shears = _gaussian_arithmetic(shears)
#     #
#     # # Now convert everything back to one affine matrix per element
#     # translations = [validate_and_reshape_matrix(setup_translation_matrix(x, 2), ndim=2) for x in translations]
#     # rotations = [validate_and_reshape_matrix(setup_2d_rotation_matrix_from_angle(x), ndim=2) for x in rotations]
#     # zooms = [validate_and_reshape_matrix(setup_scale_matrix(x, 2), ndim=2) for x in zooms]
#     # shears = [validate_and_reshape_matrix(setup_shear_matrix(x, 2), ndim=2) for x in shears]
#     # sequence = [
#     #     # np.dot(translations[idx], np.dot(rotations[idx], np.dot(zooms[idx], shears[idx])))[:2]
#     #     np.dot(np.dot(np.dot(translations[idx], rotations[idx]), zooms[idx]), shears[idx])[:2]
#     #     # np.dot(shears[idx], np.dot(zooms[idx], np.dot(rotations[idx], translations[idx])))[:2]
#     #     # np.dot(translations[idx], np.dot(rotations[idx], np.dot(shears[idx], zooms[idx])))[:2]
#     #     for idx in range(len(sequence))
#     # ]
#
#     # from ..library.elastix import save_transforms
#     # sequence = [
#     #     save_transforms(x, None, 'M', 'C', ndim=2)
#     #     for x in sequence
#     # ]
#
#     return sequence


def extract_approximate_rotation_affine(transform, coerce_affine_dimension):
    # print(transform)

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
        transform,
        fill_mode='nearest',
        cval=0.,
        order=3,
        no_offset_to_center=False,
        apply='all',
        scale_canvas=False,
        verbose=False
):

    if verbose:
        print(f'x.ndim = {x.ndim}')
        print(f'x.shape = {x.shape}')
        print(f'transform_matrix = {transform}')

    if transform.get_pivot() is None and not no_offset_to_center:
        transform.update_parameters(
            transform_matrix_offset_center(transform.get_matrix('Ms'), x.shape, ndim=x.ndim)
        )
    if transform.get_pivot() is not None:
        transform.update_parameters(
            transform_matrix_offset_center(
                transform.get_matrix('Ms'), transform.get_pivot() * 2, ndim=x.ndim
            ).flatten()[:(x.ndim + 1) * x.ndim]
        )

    if apply == 'rotation':
        raise NotImplementedError
        # TODO
        # transform_matrix = extract_approximate_rotation_affine(transform_matrix, 0)

    import scipy.ndimage as ndi
    x = ndi.affine_transform(
        x,
        transform.get_matrix('Ms'),
        order=order,
        mode=fill_mode,
        cval=cval)

    if scale_canvas:
        assert no_offset_to_center and (transform.get_pivot() == np.array([0., 0., 0.])).all()
        _, _, scale, _ = transform.decompose()
        if verbose:
            print(f'scale = {scale}')
        # This purposely rounds down, changing this to np.ceil(...) will break stuff!
        crop = np.dot(x.shape, (-scale).get_matrix('M'))[:x.ndim].astype(int)
        if verbose:
            print(f'crop = {crop}')
        if x.ndim == 3:
            x = x[:crop[0], :crop[1], :crop[2]]
        elif x.ndim == 2:
            x = x[:crop[0], :crop[1]]
        else:
            raise RuntimeError(f'Invalid number of dimensions: {x.ndim}')

    return x


# def scale_affine_matrix(affine, scale, xy_pivot):
#     # Translations are stored in pixels so need to be adjusted
#     affine[:2, 2] = np.array(affine)[:2, 2] * scale
#     # Move the pivot according to the scale
#     pivot_matrix = np.array([
#         [1., 0., xy_pivot[0]],
#         [0., 1., xy_pivot[1]],
#         [0., 0., 1.]
#     ])
#     affine = np.dot(
#         affine,
#         pivot_matrix
#     )
#     pivot_matrix[:2, 2] *= scale
#     affine = np.dot(
#         np.linalg.inv(pivot_matrix),
#         affine
#     )
#     return affine
#
#
# def scale_sequential_affines(transform_sequence, scale, xy_pivot=(0., 0.)):
#
#     transform_sequence = np.array(transform_sequence)
#
#     transform_sequence = [
#         scale_affine_matrix(transform, scale, xy_pivot)
#         for transform in transform_sequence
#     ]
#     # FIXME this really only works if the affine sequence is already sequenced (i.e. added up)
#     # print(f'scale = {scale}')
#     if scale > 1 or 1/scale - int(1/scale) != 0:
#         # z-interpolation to extend the stack
#         from scipy.ndimage import zoom
#         transform_sequence = zoom(transform_sequence, (scale, 1., 1.), order=1)
#     else:
#         transform_sequence = [
#             transform_sequence[idx]
#             for idx in range(0, len(transform_sequence), int(1/scale))
#         ]
#
#     return transform_sequence


# def sequence_affine_stack(transform_sequence, param_order='C', out_param_order='C', verbose=False):
#
#     from squirrel.library.elastix import save_transforms
#
#     result_transforms = []
#     transform = None
#
#     for this_transform in transform_sequence:
#
#         this_transform = save_transforms(
#             this_transform,
#             None,
#             param_order=param_order,
#             save_order='M',
#             ndim=2,
#             verbose=verbose
#         )
#         if verbose:
#             print(f'this_transform = {this_transform}')
#         this_transform = validate_and_reshape_matrix(
#             this_transform, ndim=2
#         )
#
#         if transform is not None:
#             transform = np.dot(this_transform, transform)
#         else:
#             transform = this_transform
#
#         result_transforms.append(
#             save_transforms(
#                 transform, None,
#                 param_order='M', save_order=out_param_order, ndim=2, verbose=verbose
#             )[:6].tolist()
#         )
#
#     return result_transforms


def apply_stack_alignment_slice(
        stack_h,
        stack_shape,
        transform,
        idx,
        n_slices=None,
        quiet=False,
        verbose=False
):

    from squirrel.library.io import get_reshaped_data

    if not quiet:
        print(f'idx = {idx} / {n_slices}')

    z_slice = get_reshaped_data(stack_h, idx, stack_shape[1:])
    return apply_affine_transform(
        z_slice, transform,
        fill_mode='constant',
        cval=0,
        verbose=verbose
    )[:stack_shape[1], :stack_shape[2]]


def apply_stack_alignment(
        stack_h,
        stack_shape,
        transform_stack,
        no_adding_of_transforms=False,
        z_range=None,
        n_workers=1,
        quiet=False,
        verbose=False
):

    if verbose:
        print(f'transform_stack.is_sequenced = {transform_stack.is_sequenced}')
    if not transform_stack.is_sequenced and not no_adding_of_transforms:
        if verbose:
            print(f'sequencing stack!')
        transform_stack = transform_stack.get_sequenced_stack()

    stack_size = np.ceil(np.array(stack_shape)).astype(int)

    result_volume = []

    from ..library.data import norm_z_range
    z_range = norm_z_range(z_range, stack_size[0])

    if n_workers == 1:

        for stack_idx, idx in enumerate(range(*z_range)):

            result_volume.append(apply_stack_alignment_slice(
                stack_h,
                stack_size,
                transform_stack[stack_idx],
                idx,
                n_slices=z_range[1],
                quiet=quiet,
                verbose=verbose
            ))

    else:

        # from multiprocessing import Pool
        # with Pool(processes=n_workers) as p:
        #     tasks = [
        #         p.apply_async(apply_stack_alignment_slice, (
        #             stack_h,
        #             stack_shape,
        #             transform_sequence[idx],
        #             idx,
        #             xy_pivot,
        #             param_order,
        #             z_range[1],
        #             verbose
        #         ))
        #         for idx in range(*z_range)
        #     ]
        #     result_volume = [task.get() for task in tasks]

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as tpe:
            tasks = [
                tpe.submit(
                    apply_stack_alignment_slice,
                    stack_h,
                    stack_size,
                    transform_stack[stack_idx],
                    idx,
                    z_range[1],
                    quiet,
                    verbose
                )
                for stack_idx, idx in enumerate(range(*z_range))
            ]
            result_volume = [task.result() for task in tasks]

    return np.array(result_volume)

