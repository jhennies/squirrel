
import numpy as np


def get_default_parameters():

    from SimpleITK import ParameterMap
    ParameterMap()
    from SimpleITK import GetDefaultParameterMap
    parameter_map = GetDefaultParameterMap(
        'affine', numberOfResolutions=2, finalGridSpacingInPhysicalUnits=8.0
    )
    parameter_map['AutomaticParameterEstimation'] = ('true',)
    parameter_map['Interpolator'] = ('BSplineInterpolator',)
    parameter_map['FixedImagePyramid'] = ('FixedRecursiveImagePyramid',)
    parameter_map['MovingImagePyramid'] = ('MovingRecursiveImagePyramid',)
    parameter_map['AutomaticScalesEstimation'] = ('true',)
    # parameter_map['ImagePyramidSchedule'] = ('8', '8', '3', '3', '1', '1')
    parameter_map['MaximumNumberOfIterations'] = ('1024',)
    # parameter_map['MaximumStepLength'] = ('4', '2', '1')
    parameter_map['ImageSampler'] = ('RandomCoordinate',)
    parameter_map['ErodeMask'] = ('true',)
    parameter_map['NumberOfSpatialSamples'] = ('1024',)
    parameter_map['NumberOfHistogramBins'] = ('48',)
    parameter_map['BSplineInterpolationOrder'] = ('3',)
    # parameter_map['ResampleInterpolator'] = ('FinalBSplineInterpolator',)
    parameter_map['NumberOfSamplesForExactGradient'] = ('1024',)
    return parameter_map


def amst_workflow(
        pre_aligned_stack,
        out_filepath,
        raw_stack=None,
        pre_align_key='data',
        pre_align_pattern='*.tif',
        transform=None,
        auto_mask_off=False,
        median_radius=4,
        z_range=None,
        elastix_parameters=get_default_parameters(),
        quiet=False,
        verbose=False
):

    if verbose:
        print(f'pre_aligned_stack = {pre_aligned_stack}')
        print(f'raw_stack = {raw_stack}')
        print(f'pre_align_key = {pre_align_key}')
        print(f'pre_align_pattern = {pre_align_pattern}')
        print(f'transform = {transform}')
        print(f'auto_mask_off={auto_mask_off}')
        print(f'median_radius = {median_radius}')
        print(f'z_range = {z_range}')

    if raw_stack is not None:
        raise NotImplementedError
    if transform != 'affine':
        raise NotImplementedError

    # Load pre-alignment

    from ..library.io import load_data_handle
    from ..library.data import norm_z_range

    handle, stack_shape = load_data_handle(pre_aligned_stack, key=pre_align_key, pattern=pre_align_pattern)

    z_range = norm_z_range(z_range, stack_shape[0])
    z_range_load = [z_range[0] - median_radius, z_range[1] + median_radius]
    z_range_load = norm_z_range(z_range_load, stack_shape[0])
    pre_align_stack = handle[z_range_load[0]: z_range_load[1]]

    # Perform the median smoothing

    from scipy.signal import medfilt
    crop = np.array(z_range) - np.array(z_range_load)
    mst = medfilt(pre_align_stack, kernel_size=[median_radius * 2 + 1, 1, 1])[crop[0]: crop[1] if crop[1] else None]
    pre_align_stack = pre_align_stack[crop[0]: crop[1] if crop[1] else None]

    from h5py import File
    with File('/media/julian/Data/tmp/mst.h5', mode='w') as f:
        f.create_dataset('data', data=mst, compression='gzip')
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
        return_result_image=False,
        pre_fix_big_jumps=False,
        parameter_map=elastix_parameters,
        quiet=quiet,
        verbose=verbose
    )

    result_transforms.to_file(out_filepath)
