import SimpleITK as sitk
import numpy as np


def apply_transform(
        image,
        transform,
        verbose=False
):
    # TODO Figure out a way to do this with Transformix
    pass


# def save_transforms(parameters, out_filepath, param_order='M', save_order='M', ndim=3, verbose=False):
#
#     parameters = np.array(parameters)
#     if verbose:
#         print(f'parameters = {parameters}')
#
#     def _elastix2m(param):
#         pr = np.zeros(param.shape, dtype=param.dtype)
#         pr[:ndim ** 2] = param[:ndim ** 2][::-1]
#         pr[ndim ** 2:] = param[ndim ** 2:][::-1]
#         param = pr
#
#         pr = np.reshape(param[: ndim ** 2], (ndim, ndim), order='C')
#         pr = np.concatenate([pr, np.array([param[ndim ** 2:]]).swapaxes(0, 1)], axis=1)
#         return pr
#
#     def _c2m(param):
#         return np.reshape(param, (ndim, ndim + 1), order='C')
#
#     def _f2m(param):
#         return np.reshape(param, (ndim, ndim + 1), order='F')
#
#     def _m2c(param):
#         return param.flatten(order='C')
#
#     def _m2f(param):
#         return param.flatten(order='F')
#
#     def _change_order(params):
#         if param_order == 'elastix':
#             params = _elastix2m(params)
#         if param_order == 'C':
#             params = _c2m(params)
#         if param_order == 'F':
#             params = _f2m(params)
#         if save_order == 'C':
#             params = _m2c(params)
#         if save_order == 'F':
#             params = _m2f(params)
#         return params
#
#     if verbose:
#         print(f'parameters.shape = {parameters.shape}')
#         print(f'parameters.ndim = {parameters.ndim}')
#
#     if param_order != save_order:
#         if (param_order != 'M' and parameters.ndim == 2) or (param_order == 'M' and parameters.ndim == 3):
#             parameters = parameters.tolist()
#             for idx, p in enumerate(parameters):
#                 parameters[idx] = _change_order(np.array(p))
#         else:
#             parameters = _change_order(parameters)
#
#     import json
#
#     if verbose:
#         print(f'parameters.shape = {parameters.shape}')
#
#     if out_filepath is not None:
#         with open(out_filepath, mode='w') as f:
#             json.dump(parameters.tolist(), f, indent=2)
#
#     return parameters


def get_affine_rotation_parameters(euler_angles):
    return sitk.Euler3DTransform((0, 0, 0), *euler_angles).GetMatrix()


def make_auto_mask(image):
    return (image > 0).astype('uint8')


def big_jump_pre_fix(moving_image, fixed_image):

    union = np.zeros(moving_image.shape, dtype=bool)
    union[moving_image > 0] = True
    union[fixed_image > 0] = True

    intersection = np.zeros(moving_image.shape, dtype=bool)
    intersection[np.logical_and(moving_image > 0, fixed_image > 0)] = True

    iou = intersection.sum() / union.sum()
    if iou < 0.5:
        print(f'Fixing big jump!')
        from skimage.registration import phase_cross_correlation
        from scipy.ndimage.interpolation import shift

        offsets = phase_cross_correlation(
            fixed_image, moving_image,
            reference_mask=fixed_image > 0,
            moving_mask=moving_image > 0,
            upsample_factor=1
        )[0]

        result_image = shift(moving_image, np.round(offsets))
        # if mask_im is not None:
        #     mask_im = shift(mask_im, np.round(offsets))

        return np.round(offsets), result_image  # , mask_im

    return (0., 0.), moving_image  # , mask_im


def register_with_elastix(
        fixed_image, moving_image,
        transform=None,
        automatic_transform_initialization=False,
        out_dir=None,
        params_to_origin=False,
        auto_mask=False,
        number_of_spatial_samples=None,
        maximum_number_of_iterations=None,
        number_of_resolutions=None,
        return_result_image=False,
        pre_fix_big_jumps=False,
        parameter_map=None,
        verbose=False
):

    if parameter_map is None:
        assert transform is not None, 'Either parameter_map or transform must be specified!'
        # Set the parameters
        parameter_map = sitk.GetDefaultParameterMap(transform if transform != 'SimilarityTransform' else 'rigid')
        parameter_map['AutomaticTransformInitialization'] = ['true' if automatic_transform_initialization else 'false']
        if number_of_spatial_samples is not None:
            parameter_map['NumberOfSpatialSamples'] = (str(number_of_spatial_samples),)
            parameter_map['NumberOfSamplesForExactGradient'] = (str(number_of_spatial_samples * 2),)
        if maximum_number_of_iterations is not None:
            parameter_map['MaximumNumberOfIterations'] = (str(maximum_number_of_iterations),)
        if number_of_resolutions is not None:
            parameter_map['NumberOfResolutions'] = (str(number_of_resolutions),)
        if transform == 'SimilarityTransform':
            parameter_map['Transform'] = ['SimilarityTransform']
    if transform is None:
        assert parameter_map is not None,  'Either parameter_map or transform must be specified!'
        transform = parameter_map['Transform'][0]
    # assert transform == parameter_map['Transform'][0]

    normalize_images = False
    if normalize_images:
        assert type(fixed_image) == np.ndarray
        assert type(moving_image) == np.ndarray
        from ..library.data import norm_8bit
        fixed_image = norm_8bit(fixed_image, (0.1, 0.9), ignore_zeros=True)
        moving_image = norm_8bit(moving_image, (0.1, 0.9), ignore_zeros=True)

    pre_fix_offsets = np.array((0., 0.))
    if pre_fix_big_jumps:
        assert type(fixed_image) == np.ndarray
        assert type(moving_image) == np.ndarray
        if transform != 'translation':
            raise NotImplementedError('Big jump fixing only implemented for translations!')
        pre_fix_offsets, moving_image = big_jump_pre_fix(moving_image, fixed_image)
        pre_fix_offsets = np.array(pre_fix_offsets)
        print(f'pre_fix_offsets: {pre_fix_offsets}')

    if type(fixed_image) == np.ndarray:
        if verbose:
            print(f'Getting fixed image from array with shape = {fixed_image.shape}')
        fixed_image = sitk.GetImageFromArray(fixed_image)
    if type(moving_image) == np.ndarray:
        if verbose:
            print(f'Getting moving image from array with shape = {moving_image.shape}')
        moving_image = sitk.GetImageFromArray(moving_image)

    if verbose:
        print(f'fixed_image.GetSize() = {fixed_image.GetSize()}')
        print(f'moving_image.GetSize() = {moving_image.GetSize()}')

    # Set the input images
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)
    if auto_mask:
        fixed_mask = make_auto_mask(sitk.GetArrayFromImage(fixed_image))
        moving_mask = make_auto_mask(sitk.GetArrayFromImage(moving_image))
        mask = fixed_mask * moving_mask
        mask = sitk.GetImageFromArray(mask)
        elastixImageFilter.SetFixedMask(mask)
        elastixImageFilter.SetMovingMask(mask)
        # parameter_map['ErodeMask'] = ['true']
    if out_dir is not None:
        elastixImageFilter.SetOutputDirectory(out_dir)
    elastixImageFilter.LogToConsoleOff()

    elastixImageFilter.SetParameterMap(parameter_map)

    if verbose:
        print(f'Running Elastix with these parameters:')
        elastixImageFilter.PrintParameterMap()

    elastixImageFilter.Execute()
    result_image = None
    if return_result_image:
        result_image = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())

    elastix_transform_param_map = elastixImageFilter.GetTransformParameterMap()[0]
    result_transform_parameters = elastix_transform_param_map['TransformParameters']
    try:
        pivot = [float(x) for x in elastix_transform_param_map['CenterOfRotationPoint']]
    except IndexError:
        pivot = [0., 0.]

    from ..library.affine_matrices import AffineMatrix
    result_matrix = AffineMatrix(
        elastix_parameters=[transform, [float(x) for x in result_transform_parameters]],
        pivot=pivot
    )
    if params_to_origin:
        if verbose:
            print(f'shifting params to origin')
            print(f'result_matrix.get_pivot() = {result_matrix.get_pivot()}')
        result_matrix.shift_pivot_to_origin()
    result_matrix = result_matrix * -AffineMatrix(parameters=[1., 0., pre_fix_offsets[0], 0., 1., pre_fix_offsets[1]])

    return result_matrix, result_image


    # if transform == 'translation':
    #
    #     from ..library.transformation import setup_translation_matrix
    #
    #     return dict(
    #         result_image=result_image,
    #         # affine_parameters=list(np.eye(result_image.ndim).flatten('C')) + list(result_transform_parameters),
    #         # affine_parameters=[[1., 0, result_transform_parameters[0]], [0., 1., result_transform_parameters[1]]],
    #         affine_parameters=setup_translation_matrix(
    #             [float(x) for x in result_transform_parameters[::-1]] - pre_fix_offsets, ndim=2
    #         ),
    #         affine_param_order='M',
    #         # translation_parameters=np.array(result_transform_parameters)
    #     )
    #
    # if transform == 'rigid':
    #
    #     assert result_image.ndim == 3, 'Rigid only implemented for volumes'
    #
    #     rotation = get_affine_rotation_parameters([float(x) for x in result_transform_parameters[:3]])
    #     affine_parameters = list(rotation) + [float(x) for x in result_transform_parameters[3:]]
    #
    #     return dict(
    #         result_image=result_image,
    #         affine_parameters=affine_parameters,
    #         affine_param_order='elastix',
    #         rigid_parameters=result_transform_parameters
    #     )
    #
    # if transform == 'SimilarityTransform':
    #
    #     assert result_image.ndim == 3, 'SimilarityTransform only implemented for volumes'
    #
    #     rotation = get_affine_rotation_parameters([float(x) for x in result_transform_parameters[:3]])
    #     affine_parameters = list(rotation) + [float(x) for x in result_transform_parameters[3:6]]
    #     affine_parameters[0] = affine_parameters[0] * float(result_transform_parameters[6])
    #     affine_parameters[3] = affine_parameters[3] * float(result_transform_parameters[6])
    #     affine_parameters[6] = affine_parameters[6] * float(result_transform_parameters[6])
    #
    #     return dict(
    #         result_image=result_image,
    #         affine_parameters=affine_parameters,
    #         affine_param_order='elastix',
    #         similarity_parameters=result_transform_parameters
    #     )
    #
    # result_transform_parameters = [float(x) for x in result_transform_parameters]
    #
    # if params_to_origin:
    #     # assert result_image.ndim == 2
    #     result_transform_parameters = save_transforms(
    #         result_transform_parameters, None,
    #         param_order='elastix', save_order='M', ndim=2, verbose=verbose
    #     )
    #     from squirrel.library.transformation import validate_and_reshape_matrix
    #     result_transform_parameters = validate_and_reshape_matrix(result_transform_parameters, 2)
    #     pivot = elastixImageFilter.GetTransformParameterMap()[0]['CenterOfRotationPoint']
    #     pivot = [float(x) for x in pivot]
    #     offset = np.array(pivot) - np.dot(result_transform_parameters[:2, :2], np.array(pivot))
    #     pivot_matrix = np.array([
    #         [1., 0., offset[0]],
    #         [0., 1., offset[1]],
    #         [0., 0., 1.]
    #     ])
    #     result_transform_parameters = np.dot(pivot_matrix, result_transform_parameters)[:2].tolist()
    #
    #     return dict(
    #         result_image=result_image,
    #         affine_parameters=result_transform_parameters,
    #         affine_param_order='M'
    #     )
    #
    # return dict(
    #     result_image=result_image,
    #     affine_parameters=result_transform_parameters,
    #     affine_param_order='elastix'
    # )


def slice_wise_stack_to_stack_alignment(
        moving_stack,
        fixed_stack,
        transform='affine',
        automatic_transform_initialization=False,
        out_dir=None,
        auto_mask=False,
        number_of_spatial_samples=None,
        maximum_number_of_iterations=None,
        number_of_resolutions=None,
        return_result_image=False,
        pre_fix_big_jumps=False,
        parameter_map=None,
        verbose=False
):

    from ..library.affine_matrices import AffineStack
    result_stack = []
    result_transforms = AffineStack(is_sequenced=True, pivot=[0., 0.])

    for zidx, z_slice_moving in enumerate(moving_stack):
        print(f'{zidx} / {len(moving_stack) - 1}')
        z_slice_fixed = fixed_stack[zidx]

        result_matrix, result_image = register_with_elastix(
            z_slice_fixed, z_slice_moving,
            transform=transform,
            automatic_transform_initialization=automatic_transform_initialization,
            out_dir=out_dir,
            auto_mask=auto_mask,
            number_of_spatial_samples=number_of_spatial_samples,
            maximum_number_of_iterations=maximum_number_of_iterations,
            number_of_resolutions=number_of_resolutions,
            return_result_image=return_result_image,
            pre_fix_big_jumps=pre_fix_big_jumps,
            parameter_map=parameter_map,
            verbose=verbose
        )
        result_matrix.shift_pivot_to_origin()

        if result_image is not None:
            result_stack.append(result_image)
        result_transforms.append(result_matrix)

    return result_transforms, result_stack


def translation_to_c(parameters):
    from ..library.transformation import setup_translation_matrix
    return setup_translation_matrix([float(x) for x in parameters[::-1]], ndim=2).flatten()


def affine_to_c(parameters):

    parameters = np.array(parameters)
    ndim = 0
    if len(parameters) == 6:
        ndim = 2
    if len(parameters) == 12:
        ndim = 3
    assert ndim in [2, 3], f'Invalid parameters: {parameters}'

    pr = np.zeros(parameters.shape, dtype=parameters.dtype)
    pr[:ndim ** 2] = parameters[:ndim ** 2][::-1]
    pr[ndim ** 2:] = parameters[ndim ** 2:][::-1]
    param = pr

    pr = np.reshape(param[: ndim ** 2], (ndim, ndim), order='C')
    pr = np.concatenate([pr, np.array([param[ndim ** 2:]]).swapaxes(0, 1)], axis=1)
    return pr.flatten()


def rigid_to_c(parameters):
    raise NotImplementedError


def similarity_to_c(parameters):
    raise NotImplementedError


def elastix_to_c(transform, parameters):
    func = None
    if transform == 'translation':
        func = translation_to_c
    if transform == 'affine':
        func = affine_to_c
    if transform == 'rigid':
        func = rigid_to_c
    if transform == 'SimilarityTransform':
        func = similarity_to_c
    if func is None:
        raise ValueError(f'Invalid transform: {transform}')
    return func(parameters)
