
import numpy as np


def get_default_parameters(transform):

    from SimpleITK import ParameterMap
    ParameterMap()
    from SimpleITK import GetDefaultParameterMap
    # parameter_map = GetDefaultParameterMap(
    #     'affine', numberOfResolutions=2, finalGridSpacingInPhysicalUnits=8.0
    # )
    # parameter_map['AutomaticParameterEstimation'] = ('true',)
    # parameter_map['Interpolator'] = ('BSplineInterpolator',)
    # parameter_map['FixedImagePyramid'] = ('FixedRecursiveImagePyramid',)
    # parameter_map['MovingImagePyramid'] = ('MovingRecursiveImagePyramid',)
    # parameter_map['AutomaticScalesEstimation'] = ('true',)
    # # parameter_map['ImagePyramidSchedule'] = ('8', '8', '3', '3', '1', '1')
    # parameter_map['MaximumNumberOfIterations'] = ('1024',)
    # # parameter_map['MaximumStepLength'] = ('4', '2', '1')
    # parameter_map['ImageSampler'] = ('RandomCoordinate',)
    # parameter_map['ErodeMask'] = ('true',)
    # parameter_map['NumberOfSpatialSamples'] = ('1024',)
    # parameter_map['NumberOfHistogramBins'] = ('48',)
    # parameter_map['BSplineInterpolationOrder'] = ('3',)
    # # parameter_map['ResampleInterpolator'] = ('FinalBSplineInterpolator',)
    # parameter_map['NumberOfSamplesForExactGradient'] = ('1024',)
    if transform == 'affine':
        parameter_map = GetDefaultParameterMap(
            transform, numberOfResolutions=4, finalGridSpacingInPhysicalUnits=8.0
        )
        parameter_map['AutomaticParameterEstimation'] = ('true',)
        parameter_map['Interpolator'] = ('BSplineInterpolator',)
        parameter_map['ResampleInterpolator'] = ('FinalBSplineInterpolator',)
        parameter_map['FixedImagePyramid'] = ('FixedRecursiveImagePyramid',)
        parameter_map['MovingImagePyramid'] = ('MovingRecursiveImagePyramid',)
        parameter_map['AutomaticScalesEstimation'] = ('false',)
        # # parameter_map['ImagePyramidSchedule'] = ('8', '8', '3', '3', '1', '1')
        parameter_map['MaximumNumberOfIterations'] = ('256',)
        # # parameter_map['MaximumStepLength'] = ('4', '2', '1')
        # parameter_map['ImageSampler'] = ('RandomCoordinate',)
        parameter_map['ErodeMask'] = ('true',)
        # parameter_map['NumberOfSpatialSamples'] = ('2048',)
        parameter_map['NumberOfHistogramBins'] = ('32',)
        parameter_map['BSplineInterpolationOrder'] = ('1',)
        # parameter_map['NumberOfSamplesForExactGradient'] = ('1024',)
        return parameter_map
    if transform == 'bspline':
        parameter_map = GetDefaultParameterMap('bspline')

        parameter_map['AutomaticParameterEstimation'] = ["true"]
        parameter_map['CheckNumberOfSamples'] = ["true"]
        parameter_map['DefaultPixelValue'] = ["0.000000"]
        parameter_map['FinalBSplineInterpolationOrder'] = ["3.000000"]
        parameter_map['FinalGridSpacingInPhysicalUnits'] = ["512.000000"]
        parameter_map['FixedImagePyramid'] = ["FixedRecursiveImagePyramid"]
        parameter_map['GridSpacingSchedule'] = ["2.803221", "1.988100", "1.410000", "1.000000"]
        parameter_map['ImageSampler'] = ["RandomCoordinate"]
        parameter_map['Interpolator'] = ["BSplineInterpolator"]
        parameter_map['MaximumNumberOfIterations'] = ["512.000000"]
        parameter_map['MaximumNumberOfSamplingAttempts'] = ["8.000000"]
        parameter_map['Metric'] = ["AdvancedMattesMutualInformation"]
        parameter_map['MovingImagePyramid'] = ["MovingRecursiveImagePyramid"]
        parameter_map['NewSamplesEveryIteration'] = ["true"]
        parameter_map['NumberOfResolutions'] = ["4.000000"]
        parameter_map['NumberOfSamplesForExactGradient'] = ["4096.000000"]
        parameter_map['NumberOfSpatialSamples'] = ["2048.000000"]
        parameter_map['Optimizer'] = ["AdaptiveStochasticGradientDescent"]
        parameter_map['Registration'] = ["MultiResolutionRegistration"]
        parameter_map['ResampleInterpolator'] = ["FinalBSplineInterpolator"]
        parameter_map['Resampler'] = ["DefaultResampler"]
        parameter_map['ResultImageFormat'] = ["nii"]
        parameter_map['Transform'] = ["BSplineTransform"]
        parameter_map['WriteIterationInfo'] = ["false"]
        parameter_map['WriteResultImage'] = ["false"]
        parameter_map['ErodeMask'] = ["true"]
        parameter_map['NumberOfHistogramBins'] = ["32"]
        parameter_map['AutomaticScalesEstimation'] = ["false"]

        return parameter_map


def _z_smooth(
        inp,
        median_radius,
        method
):
    if method == 'median':
        from scipy.signal import medfilt
        return medfilt(inp, kernel_size=[median_radius * 2 + 1, 1, 1])

    if method == 'gaussian':
        from vigra.filters import gaussianSmoothing
        dtype = inp.dtype
        return gaussianSmoothing(inp.astype('float32'), [median_radius, 0, 0]).astype(dtype)


# from h5py import File
# with File('/media/julian/Data/tmp/raw.h5', mode='r') as f:
#     data = f['data'][128:-128, 128:-128, 128:-128]
#
# mst_gauss = _z_smooth(data, 7, 'gaussian')
# with File('/media/julian/Data/tmp/mst_gauss.h5', mode='w') as f:
#     f.create_dataset('data', data=mst_gauss)
#
# mst_median = _z_smooth(data, 7, 'median')
# with File('/media/julian/Data/tmp/mst_median.h5', mode='w') as f:
#     f.create_dataset('data', data=mst_median)


def amst_workflow(
        pre_aligned_stack,
        out_filepath,
        raw_stack=None,
        pre_align_key='data',
        pre_align_pattern='*.tif',
        transform=None,
        auto_mask_off=False,
        median_radius=7,
        z_smooth_method='median',
        z_range=None,
        gaussian_sigma=0.,
        elastix_parameters=None,
        crop_to_bounds_off=False,
        quiet=False,
        debug=False,
        verbose=False
):

    if elastix_parameters is None:
        elastix_parameters = get_default_parameters(transform)
    elif type(elastix_parameters) == str:
        from SimpleITK import ReadParameterFile
        elastix_parameters = ReadParameterFile(elastix_parameters)
    transform = elastix_parameters['Transform'][0]

    if verbose:
        print(f'pre_aligned_stack = {pre_aligned_stack}')
        print(f'raw_stack = {raw_stack}')
        print(f'pre_align_key = {pre_align_key}')
        print(f'pre_align_pattern = {pre_align_pattern}')
        print(f'transform = {transform}')
        print(f'auto_mask_off={auto_mask_off}')
        print(f'median_radius = {median_radius}')
        print(f'z_range = {z_range}')
        print(f'gaussian_sigma = {gaussian_sigma}')

    if transform == 'BSplineTransform':
        transform = 'bspline'
    elif transform == 'AffineTransform':
        transform = 'affine'
    else:
        raise ValueError(f'Invalid transform for AMST workflow: {transform}')

    if raw_stack is not None:
        raise NotImplementedError
    if transform not in ['affine', 'bspline']:
        raise NotImplementedError(f'Not implemented for transform = {transform}')

    # Load pre-alignment

    from ..library.io import load_data_handle
    from ..library.data import norm_z_range

    handle, stack_shape = load_data_handle(pre_aligned_stack, key=pre_align_key, pattern=pre_align_pattern)

    z_range = norm_z_range(z_range, stack_shape[0])
    z_range_load = [z_range[0] - median_radius, z_range[1] + median_radius]
    z_range_load = norm_z_range(z_range_load, stack_shape[0])
    pre_align_stack = handle[z_range_load[0]: z_range_load[1]]

    # Perform the median smoothing

    # from scipy.signal import medfilt
    crop = np.array(z_range) - np.array(z_range_load)
    # mst = medfilt(pre_align_stack, kernel_size=[median_radius * 2 + 1, 1, 1])[crop[0]: crop[1] if crop[1] else None]
    mst = _z_smooth(pre_align_stack, median_radius=median_radius, method=z_smooth_method)[crop[0]: crop[1] if crop[1] else None]
    pre_align_stack = pre_align_stack[crop[0]: crop[1] if crop[1] else None]

    # from h5py import File
    # with File('/media/julian/Data/tmp/mst.h5', mode='w') as f:
    #     f.create_dataset('data', data=mst, compression='gzip')
    # assert mst.shape == stack_shape, f'mst.shape = {mst.shape}; stack_shape = {stack_shape}'
    if verbose:
        print(f'pre_align_stack.shape = {pre_align_stack.shape}')
        print(f'mst.shape = {mst.shape}')

    # Alignment to median smoothed template

    from ..library.elastix import slice_wise_stack_to_stack_alignment
    result_transforms, _ = slice_wise_stack_to_stack_alignment(
        pre_align_stack, mst,
        transform=transform,
        automatic_transform_initialization=False,
        out_dir=None,
        auto_mask=not auto_mask_off,
        return_result_image=True,
        pre_fix_big_jumps=False,
        parameter_map=elastix_parameters,
        gaussian_sigma=gaussian_sigma,
        crop_to_bounds_off=crop_to_bounds_off,
        normalize_images=False,  # Do not normalize since the images come from the same source
        quiet=quiet,
        debug=debug,
        verbose=verbose
    )

    # from h5py import File
    # with File('/media/julian/Data/projects/hennies/amst_devel/amst_stack_elastic_ref.h5', mode='w') as f:
    #     f.create_dataset('data', data=np.clip(np.array(result_stack), 0, 255).astype('uint8'), compression='gzip')

    result_transforms.to_file(out_filepath)
