import os

import numpy as np


def apply_transforms_on_image(
        image,
        transforms,
        n_workers=os.cpu_count(),
        verbose=False
):

    import SimpleITK as sitk

    txif = sitk.TransformixImageFilter()
    for transform in transforms[::-1]:
        txif.AddTransformParameterMap(transform)
    txif.SetMovingImage(sitk.GetImageFromArray(image))
    txif.LogToConsoleOn()
    txif.Execute()
    result_final = sitk.GetArrayFromImage(txif.GetResultImage())
    return result_final

    # result = image
    # for transform in transforms:
    #     print(f'result.shape = {result.shape}')
    #     txif = sitk.TransformixImageFilter()
    #     txif.AddTransformParameterMap(transform)
    #     txif.SetMovingImage(sitk.GetImageFromArray(result))
    #     txif.LogToConsoleOff()
    #     txif.Execute()
    #     result = sitk.GetArrayFromImage(txif.GetResultImage())
    #
    # return result


def apply_transforms_on_image_stack_slice(
        image_stack_h,
        image_idx,
        transforms,
        target_image_shape=None,
        n_slices=None,
        n_workers=os.cpu_count(),
        quiet=False,
        verbose=False
):
    target_image_shape = image_stack_h[0].shape if target_image_shape is None else target_image_shape
    if not quiet or verbose:
        print(f'image_idx = {image_idx} / {n_slices}')

    from squirrel.library.io import get_reshaped_data
    z_slice = get_reshaped_data(image_stack_h, image_idx, target_image_shape)

    return apply_transforms_on_image(
        z_slice,
        transforms,
        n_workers=n_workers
    )[:target_image_shape[0], :target_image_shape[1]]

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


def make_auto_mask(image, disk_size=6, method='non-zero', variance_filter_size=3, variance_thresh=10):
    from skimage.morphology import binary_closing, disk
    from squirrel.library.scaling import scale_image_nearest

    mask = None
    if method == 'non-zero':
        mask = (image > 0).astype('uint8')
    if method == 'variance':
        from scipy.ndimage import generic_filter
        mask_t = (generic_filter(scale_image_nearest(image, scale_factors=[0.25, 0.25]), np.var, size=variance_filter_size) > variance_thresh).astype('uint8')
        mask_t = scale_image_nearest(mask_t, scale_factors=[4, 4])
        mask = np.zeros(image.shape, dtype='uint8')
        mask[:mask_t.shape[0], :mask_t.shape[1]] = mask_t
        mask[image == 0] = 0
    if mask is None:
        raise ValueError(f'Invalid auto-mask method: {method}')

    footprint = disk(disk_size)
    mask = binary_closing(mask, footprint).astype('uint8')

    return mask


def big_jump_pre_fix(moving_orig, fixed_orig, moving_image, fixed_image, iou_thresh=0.5, verbose=False):

    union = np.zeros(moving_orig.shape, dtype=bool)
    union[moving_orig > 0] = True
    union[fixed_orig > 0] = True

    intersection = np.zeros(moving_orig.shape, dtype=bool)
    intersection[np.logical_and(moving_orig > 0, fixed_orig > 0)] = True

    iou = intersection.sum() / union.sum()
    print(f'IoU = {iou}')
    if iou < iou_thresh:
        print(f'Fixing big jump! (IoU = {iou} < {iou_thresh})')
        from skimage.registration import phase_cross_correlation
        from scipy.ndimage.interpolation import shift

        offsets = phase_cross_correlation(
            fixed_image, moving_image,
            reference_mask=fixed_image > 0,
            moving_mask=moving_image > 0,
            upsample_factor=1,
            normalization=None
        )[0]

        result_image = shift(moving_image, np.round(offsets))

        if verbose:
            print(f'offsets = {offsets}')

        return np.round(offsets), result_image

    return (0., 0.), moving_image


def register_with_elastix(
        fixed_image, moving_image,
        transform=None,
        automatic_transform_initialization=False,
        out_dir=None,
        params_to_origin=False,
        auto_mask=None,
        number_of_spatial_samples=None,
        maximum_number_of_iterations=None,
        number_of_resolutions=None,
        return_result_image=False,
        pre_fix_big_jumps=False,
        pre_fix_iou_thresh=0.5,
        parameter_map=None,
        median_radius=0,
        gaussian_sigma=0.,
        use_edges=False,
        use_clahe=False,
        crop_to_bounds_off=False,
        n_workers=os.cpu_count(),
        normalize_images=True,
        debug_dirpath=None,
        verbose=False
):

    if verbose:
        print('Running register_with_elastix with:')
        print(f'automatic_transform_initialization={automatic_transform_initialization}')
        print(f'out_dir={out_dir}')
        print(f'params_to_origin={params_to_origin}')
        print(f'auto_mask={auto_mask}')
        print(f'number_of_spatial_samples={number_of_spatial_samples}')
        print(f'maximum_number_of_iterations={maximum_number_of_iterations}')
        print(f'number_of_resolutions={number_of_resolutions}')
        print(f'return_result_image={return_result_image}')
        print(f'pre_fix_big_jumps={pre_fix_big_jumps}')
        print(f'pre_fix_iou_thresh={pre_fix_iou_thresh}')
        print(f'parameter_map={parameter_map}')
        print(f'median_radius={median_radius}')
        print(f'gaussian_sigma={gaussian_sigma}')
        print(f'use_edges={use_edges}')
        print(f'use_clahe={use_clahe}')
        print(f'crop_to_bounds_off={crop_to_bounds_off}')
        print(f'n_workers={n_workers}')
        print(f'normalize_images={normalize_images}')
        print(f'debug_dirpath = {debug_dirpath}')

    if debug_dirpath is not None and not os.path.exists(debug_dirpath):
        os.makedirs(debug_dirpath, exist_ok=True)
        from tifffile import imwrite, imsave
        imwrite(os.path.join(debug_dirpath, '00-fixed.tif'), fixed_image)
        imwrite(os.path.join(debug_dirpath, '00-moving.tif'), moving_image)

    import SimpleITK as sitk

    # TODO: Properly expose crop_to_bounds independently of auto-masking
    crop_to_bounds = auto_mask is not None
    if crop_to_bounds_off:
        crop_to_bounds = False
    if transform == 'bspline':
        crop_to_bounds = False

    dtype = fixed_image.dtype
    assert dtype == 'uint8' or dtype == 'uint16', \
        f'Only allowing 8 or 16 bit unsigned integer images. Fixed image has dtype = {dtype}'
    assert moving_image.dtype == dtype, \
        f'fixed and moving images must have the same data type: {dtype} != {moving_image.dtype}'

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
    if type(parameter_map) == str:
        from SimpleITK import ReadParameterFile
        assert os.path.exists(parameter_map)
        parameter_map = ReadParameterFile(parameter_map)
    if transform is None:
        assert parameter_map is not None,  'Either parameter_map or transform must be specified!'
        transform = parameter_map['Transform'][0]
    if debug_dirpath is not None:
        from SimpleITK import WriteParameterFile
        WriteParameterFile(parameter_map, os.path.join(debug_dirpath, 'elastix_parameters.txt'))
    # assert transform == parameter_map['Transform'][0]

    mask = None
    bounds_offset = np.array([0., 0.])
    if crop_to_bounds:
        if verbose:
            print(f'image shape before auto_mask: {fixed_image.shape}')
        from squirrel.library.image import get_bounds
        bounds_fixed = get_bounds(fixed_image, return_ints=True)
        bounds_moving = get_bounds(moving_image, return_ints=True)
        bounds_total = np.array([
            min(bounds_moving[0], bounds_fixed[0]),
            min(bounds_moving[1], bounds_fixed[1]),
            max(bounds_moving[2], bounds_fixed[2]),
            max(bounds_moving[3], bounds_fixed[3])
        ]).astype(int)
        bounds_offset = bounds_total[:2]
        bounds = np.s_[bounds_total[0]: bounds_total[2], bounds_total[1]: bounds_total[3]]
        fixed_image = fixed_image[bounds]
        moving_image = moving_image[bounds]

    if debug_dirpath is not None:
        imwrite(os.path.join(debug_dirpath, '01-fixed-cropped.tif'), fixed_image)
        imwrite(os.path.join(debug_dirpath, '01-moving-cropped.tif'), moving_image)

    moving_orig = None
    fixed_orig = None
    if pre_fix_big_jumps:
        moving_orig = moving_image.copy()
        fixed_orig = fixed_image.copy()

    if use_clahe:
        from squirrel.library.normalization import clahe_on_image
        fixed_image = clahe_on_image(fixed_image)
        moving_image = clahe_on_image(moving_image)
    if median_radius > 0:
        from skimage.filters import median
        from skimage.morphology import disk
        fixed_image = median(fixed_image, footprint=disk(median_radius))
        moving_image = median(moving_image, footprint=disk(median_radius))
    if gaussian_sigma > 0:
        from skimage.filters import gaussian
        fixed_image = gaussian(fixed_image.astype(float), gaussian_sigma).astype(dtype)
        moving_image = gaussian(moving_image.astype(float), gaussian_sigma).astype(dtype)
    if use_edges:
        from skimage.filters import sobel
        fixed_image = sobel(fixed_image.astype(float))
        moving_image = sobel(moving_image.astype(float))

    if auto_mask is not None:
        fixed_mask = make_auto_mask(fixed_image, disk_size=6, method=auto_mask)
        moving_mask = make_auto_mask(moving_image, disk_size=6, method=auto_mask)
        mask = fixed_mask * moving_mask
        if debug_dirpath is not None:
            imwrite(os.path.join(debug_dirpath, '02-mask-from-fixed.tif'), fixed_mask)
            imwrite(os.path.join(debug_dirpath, '02-mask-from-moving.tif'), moving_mask)
            imwrite(os.path.join(debug_dirpath, '02-mask.tif'), mask)
        if gaussian_sigma > 0:
            from skimage.morphology import erosion
            from skimage.morphology import disk
            import math
            mask = erosion(mask, footprint=disk(int(math.ceil(3 * gaussian_sigma))))
        if use_edges:
            fixed_image[mask == 0] = 0
            moving_image[mask == 0] = 0
        if verbose:
            print(f'image shape after auto_mask: {fixed_image.shape}')

    if debug_dirpath is not None:
        imwrite(os.path.join(debug_dirpath, '03-fixed-pre-processed.tif'), fixed_image)
        imwrite(os.path.join(debug_dirpath, '03-moving-pre-processed.tif'), moving_image)
        if mask is not None:
            imwrite(os.path.join(debug_dirpath, '03-mask-pre-processed.tif'), mask)

    if normalize_images:
        assert type(fixed_image) == np.ndarray
        assert type(moving_image) == np.ndarray
        from squirrel.library.data import norm_full_range
        fixed_image = norm_full_range(fixed_image, (0.05, 0.95), ignore_zeros=False, mask=mask, cast_8bit=True)
        moving_image = norm_full_range(moving_image, (0.05, 0.95), ignore_zeros=False, mask=mask, cast_8bit=True)

    if debug_dirpath is not None:
        imwrite(os.path.join(debug_dirpath, '04-fixed-normalized.tif'), fixed_image)
        imwrite(os.path.join(debug_dirpath, '04-moving-normalized.tif'), moving_image)
        if mask is not None:
            imwrite(os.path.join(debug_dirpath, '04-mask-normalized.tif'), mask)

    pre_fix_offsets = np.array((0., 0.))
    if verbose:
        print(f'Checking whether to run pre-fix for big jumps...')
    if pre_fix_big_jumps:
        if verbose:
            print(f'Running pre-fix for big jumps!')
        assert type(fixed_image) == np.ndarray
        assert type(moving_image) == np.ndarray
        if transform != 'translation':
            raise NotImplementedError('Big jump fixing only implemented for translations!')
        pre_fix_offsets, moving_image = big_jump_pre_fix(
            moving_orig, fixed_orig,
            moving_image, fixed_image,
            iou_thresh=pre_fix_iou_thresh, verbose=verbose
        )
        pre_fix_offsets = np.array(pre_fix_offsets)

    if debug_dirpath is not None:
        imwrite(os.path.join(debug_dirpath, '03-moving-after-pre-fix.tif'), moving_image)

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
    if mask is not None:
        # fixed_mask = make_auto_mask(sitk.GetArrayFromImage(fixed_image))
        # moving_mask = make_auto_mask(sitk.GetArrayFromImage(moving_image))
        # mask = fixed_mask * moving_mask
        # from h5py import File
        # with File('/media/julian/Data/tmp/mask.h5', mode='w') as f:
        #     f.create_dataset('data', data=mask, compression='gzip')
        mask = sitk.GetImageFromArray(mask)
        elastixImageFilter.SetFixedMask(mask)
        elastixImageFilter.SetMovingMask(mask)
        parameter_map['ErodeMask'] = ['true']
    if out_dir is not None:
        elastixImageFilter.SetOutputDirectory(out_dir)
    elastixImageFilter.LogToConsoleOff()
    if verbose:
        elastixImageFilter.LogToConsoleOn()

    elastixImageFilter.SetParameterMap(parameter_map)

    if verbose:
        print(f'Running Elastix with these parameters:')
        elastixImageFilter.PrintParameterMap()
    elastixImageFilter.SetNumberOfThreads(n_workers)
    elastixImageFilter.Execute()
    result_image = None
    if return_result_image:
        result_image = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())

    elastix_transform_param_map = elastixImageFilter.GetTransformParameterMap()[0]

    # Return an affine matrix object for any rigid or affine transformation
    if transform != 'bspline':
        result_transform_parameters = elastix_transform_param_map['TransformParameters']
        try:
            pivot = np.array([float(x) for x in elastix_transform_param_map['CenterOfRotationPoint']])[::-1]
        except IndexError:
            pivot = np.array([0., 0.])
        pivot += bounds_offset

        from ..library.affine_matrices import AffineMatrix
        result_matrix = AffineMatrix(
            elastix_parameters=[transform, [float(x) for x in result_transform_parameters]]
        )
        result_matrix = result_matrix * -AffineMatrix(parameters=[1., 0., pre_fix_offsets[0], 0., 1., pre_fix_offsets[1]])
        result_matrix.set_pivot(pivot)
        # result_matrix = result_matrix * AffineMatrix(parameters=[1., 0., bounds_offset[0], 0., 1., bounds_offset[1]])

        if params_to_origin:
            if verbose:
                print(f'shifting params to origin')
                print(f'result_matrix.get_pivot() = {result_matrix.get_pivot()}')
            result_matrix.shift_pivot_to_origin()

        return result_matrix, result_image

    # Return the elastix parameters for non-rigid registration
    return elastix_transform_param_map, result_image

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
        median_radius=0,
        gaussian_sigma=0.,
        number_of_spatial_samples=None,
        maximum_number_of_iterations=None,
        number_of_resolutions=None,
        return_result_image=False,
        pre_fix_big_jumps=False,
        parameter_map=None,
        crop_to_bounds_off=False,
        normalize_images=False,
        quiet=False,
        verbose=False
):

    if transform in ['translation', 'affine']:
        from ..library.affine_matrices import AffineStack
        result_transforms = AffineStack(is_sequenced=True, pivot=[0., 0.])
    elif transform == 'bspline':
        result_transforms = ElastixStack()
    else:
        raise ValueError(f'Invalid transform: {transform}')
    result_stack = []

    for zidx, z_slice_moving in enumerate(moving_stack):
        if not quiet:
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
            median_radius=median_radius,
            gaussian_sigma=gaussian_sigma,
            crop_to_bounds_off=crop_to_bounds_off,
            normalize_images=normalize_images,
            verbose=verbose
        )
        if transform != 'bspline':
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

def c_to_elastix(parameters):
    parameters = np.array(parameters)

    ndim = 0
    if len(parameters) == 6:
        ndim = 2
    if len(parameters) == 12:
        ndim = 3
    assert ndim in [2, 3], f'Invalid parameters: {parameters}'

    out_parameters = np.zeros(parameters.shape, dtype=parameters.dtype)

    pr = np.array([parameters[((ndim + 1) * idx): ((ndim + 1) * (idx + 1))] for idx in range(ndim)])
    out_parameters[: ndim ** 2] = pr[:, :ndim].flatten()[::-1]
    out_parameters[ndim ** 2:] = pr[:, ndim][::-1]

    return out_parameters


class ElastixStack:

    def __init__(
            self,
            stack=None,
            dirpath=None,
            pattern='*.txt',
            image_shape=None
    ):
        self._dirpath = None
        self._pattern = None
        self._stack = None
        self._it = 0
        if stack is not None:
            self.set_from_stack(stack, image_shape=image_shape)
        if dirpath is not None:
            self.set_from_dir(dirpath, pattern)

    def set_from_stack(self, stack, image_shape=None):
        # assert type(stack) is list or type(stack) is tuple, 'Only accepting lists or tuples'
        from SimpleITK import ParameterMap
        if isinstance(stack[0], ParameterMap):
            self._stack = list(stack)
            return
        from squirrel.library.affine_matrices import AffineStack
        if isinstance(stack, AffineStack):
            self._stack = [
                x.to_elastix_affine(return_parameter_map=True, shape=image_shape) for x in stack
            ]

    def set_from_dir(self, dirpath, pattern):
        from SimpleITK import ReadParameterFile
        import glob
        filepaths = sorted(glob.glob(os.path.join(dirpath, pattern)))
        self._stack = [ReadParameterFile(filepath) for filepath in filepaths]
        self._dirpath = dirpath
        self._pattern = pattern

    def to_file(self, dirpath):
        from SimpleITK import WriteParameterFile
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        for idx, transform in enumerate(self._stack):
            filepath = os.path.join(dirpath, 'transform_{:05d}.txt'.format(idx))
            WriteParameterFile(transform, filepath)

    def __getitem__(self, item):
        if self._stack is not None:
            return self._stack[item]
        return []

    def __setitem__(self, key, value):
        self._stack[key] = value

    def __len__(self):
        if self._stack is None:
            return 0
        return len(self._stack)

    def __iter__(self):
        self._it = 0
        return self

    def __next__(self):
        if self._it < len(self):
            x = self[self._it]
            self._it += 1
            return x
        else:
            raise StopIteration

    def append(self, other):
        if isinstance(other, ElastixStack):
            for x in other:
                self.append(x)
            return
        from SimpleITK import ParameterMap
        if isinstance(other, ParameterMap):
            self.set_from_stack(self[:] + [other])
            return
        raise ValueError(f'Invalid type of other: {type(other)}')

    def image_shape(self):
        shapes = []
        for transform in self:
            shapes.append(np.array([int(float(x)) for x in transform['Size']])[::-1])
        shapes = np.array(shapes)
        assert np.all(shapes == shapes[0], axis=0).all(), 'Not all shapes of the transforms are equal - but they should be!'

        return shapes[0]


def load_transform_stack_from_multiple_files(paths, sequence_stack=False):

    stack = ElastixStack(dirpath=paths[0])
    for path in paths[1:]:
        stack.append(ElastixStack(dirpath=path))
    return stack


class ElastixMultiStepStack:

    def __init__(
            self,
            stacks=None,
            image_shape=None
    ):
        self._stacks = None
        self._num_steps = 0
        self._it = 0
        if stacks is not None:
            [self.add_stack(stack, image_shape=image_shape) for stack in stacks]

    def add_stack(self, stack, image_shape=None):
        stack = ElastixStack(stack=stack, image_shape=image_shape)
        if self._stacks is not None:
            assert len(stack) == len(self), 'The length of the added stack does not match'
            self._stacks = [self[idx] + [s] for idx, s in enumerate(stack)]
            self._num_steps += len(stack)
            return
        self._stacks = [[s] for s in stack]
        self._num_steps = len(stack)

    def __len__(self):
        return len(self[:])

    def __getitem__(self, item):
        if self._stacks is None:
            return []
        return self._stacks[item]

    def __setitem__(self, key, value):
        assert len(value) == self._num_steps, 'Number of elements in value does not match number of steps!'
        self._stacks[key] = value

    def __iter__(self):
        self._it = 0
        return self

    def __next__(self):
        if self._it < len(self):
            x = self[self._it]
            self._it += 1
            return x
        else:
            raise StopIteration

    def apply_on_image_stack(
            self,
            image_stack_h,
            target_image_shape=None,
            z_range=None,
            n_workers=1,
            quiet=False,
            verbose=False
    ):
        from squirrel.library.data import norm_z_range
        z_range = norm_z_range(z_range, len(image_stack_h))
        result_volume = []
        dtype = image_stack_h.dtype
        max_val = np.iinfo(dtype).max
        assert dtype == 'uint8' or dtype == 'uint16'

        if verbose:
            print(f'target_image_shape = {target_image_shape}')

        # if n_workers == 1:
        #
        for stack_idx, image_idx in enumerate(range(*z_range)):

            result_volume.append(apply_transforms_on_image_stack_slice(
                image_stack_h,
                image_idx,
                self[stack_idx],
                target_image_shape=target_image_shape,
                n_slices=z_range[1],
                n_workers=1,
                quiet=quiet,
                verbose=verbose
            ))

        if verbose:
            print(f'result_volume[0].shape = {result_volume[0].shape}')

        # else:
        #
        #     from concurrent.futures import ThreadPoolExecutor
        #     with ThreadPoolExecutor(max_workers=n_workers) as tpe:
        #         tasks = [
        #             tpe.submit(
        #                 apply_transforms_on_image_stack_slice,
        #                 image_stack_h,
        #                 image_idx,
        #                 self[stack_idx],
        #                 target_image_shape,
        #                 z_range[1],
        #                 1,
        #                 quiet,
        #                 verbose
        #             )
        #             for stack_idx, image_idx in enumerate(range(*z_range))
        #         ]
        #         result_volume = [task.result() for task in tasks]

        return np.clip(np.array(result_volume), 0, max_val).astype(dtype)


if __name__ == '__main__':
    # a = [1, 2, 3, 4, 5, 6]
    # print(f'a = {a}')
    # b = affine_to_c(a)
    # print(f'b = {b}')
    # c = c_to_elastix(b)
    # print(f'c = {c}')
    #
    # a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # print(f'a = {a}')
    # b = affine_to_c(a)
    # print(f'b = {b}')
    # c = c_to_elastix(b)
    # print(f'c = {c}')

    es = ElastixStack(dirpath='/media/julian/Data/projects/hennies/amst_devel/240624_snamst_kors_dT/amst.meta/amst/')
    print(es.image_shape())

