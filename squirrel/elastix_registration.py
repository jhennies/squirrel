
def register_z_chunks():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs Simple Elastix on chunks along the z-axis of a dataset individually.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('moving_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('fixed_filepath', type=str,
                        help='Input filepath for the fixed volume (nii or h5)')
    parser.add_argument('out_path', type=str,
                        help='Output folder for all results files (must be empty or not existing)')
    parser.add_argument('--moving_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--fixed_key', type=str, default='data',
                        help='Internal path of the fixed input; default="data"; used if fixed_filepath is h5 file')
    parser.add_argument('--transform', type=str, default='affine',
                        help='The transformation. Can be "affine" (default) or "rigid" ')
    parser.add_argument('--z_chunk_size', type=int, default=16,
                        help='Size of the z-chunks')
    parser.add_argument('-auto_init', '--automatic_transform_initialization', action='store_true',
                        help='Triggers the automatic transform initialization from Elastix; Not needed if both images'
                             'are already close')
    parser.add_argument('-napari', '--view_results_in_napari', action='store_true',
                        help='Show the results in the napari viewer')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    moving_filepath = args.moving_filepath
    fixed_filepath = args.fixed_filepath
    out_path = args.out_path
    moving_key = args.moving_key
    fixed_key = args.fixed_key
    z_chunk_size = args.z_chunk_size
    transform = args.transform
    automatic_transform_initialization = args.automatic_transform_initialization
    view_results_in_napari = args.view_results_in_napari
    verbose = args.verbose

    from squirrel.workflows.elastix import register_z_chunks

    register_z_chunks(
        moving_filepath,
        fixed_filepath,
        out_path,
        moving_key=moving_key,
        fixed_key=fixed_key,
        z_chunk_size=z_chunk_size,
        transform=transform,
        automatic_transform_initialization=automatic_transform_initialization,
        view_results_in_napari=view_results_in_napari,
        verbose=verbose
    )


def elastix_on_volume3d():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs Simple Elastix with an affine transform',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('moving_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('fixed_filepath', type=str,
                        help='Input filepath for the fixed volume (nii or h5)')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for all results files (must be empty or not existing)')
    parser.add_argument('--moving_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--fixed_key', type=str, default='data',
                        help='Internal path of the fixed input; default="data"; used if fixed_filepath is h5 file')
    parser.add_argument('--transform', type=str, default='affine',
                        help='The transformation; default="affine"')
    parser.add_argument('-auto_init', '--automatic_transform_initialization', action='store_true',
                        help='Triggers the automatic transform initialization from Elastix; Not needed if both images'
                             'are already close')
    parser.add_argument('--pivot', type=float, nargs=3, default=(0., 0., 0.),
                        help='To where the resulting transformation is centered')
    parser.add_argument('-napari', '--view_results_in_napari', action='store_true',
                        help='Show the results in the napari viewer')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    moving_filepath = args.moving_filepath
    fixed_filepath = args.fixed_filepath
    out_filepath = args.out_filepath
    moving_key = args.moving_key
    fixed_key = args.fixed_key
    transform = args.transform
    automatic_transform_initialization = args.automatic_transform_initialization
    pivot = args.pivot
    view_results_in_napari = args.view_results_in_napari
    verbose = args.verbose

    from squirrel.workflows.elastix import elastix3d

    elastix3d(
        moving_filepath,
        fixed_filepath,
        out_filepath,
        moving_key=moving_key,
        fixed_key=fixed_key,
        transform=transform,
        automatic_transform_initialization=automatic_transform_initialization,
        pivot=pivot,
        view_results_in_napari=view_results_in_napari,
        verbose=verbose
    )


def elastix_slices_to_volume():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Register 2D slices of an image stack to a corresponding volume',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('moving_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('fixed_filepath', type=str,
                        help='Input filepath for the fixed volume (nii or h5)')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the result volume.')
    parser.add_argument('--moving_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--fixed_key', type=str, default='data',
                        help='Internal path of the fixed input; default="data"; used if fixed_filepath is h5 file')
    parser.add_argument('--transform', type=str, default='affine',
                        help='The transformation; default="affine"')
    parser.add_argument('-auto_init', '--automatic_transform_initialization', action='store_true',
                        help='Triggers the automatic transform initialization from Elastix; Not needed if both images'
                             'are already close')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    moving_filepath = args.moving_filepath
    fixed_filepath = args.fixed_filepath
    out_filepath = args.out_filepath
    moving_key = args.moving_key
    fixed_key = args.fixed_key
    transform = args.transform
    automatic_transform_initialization = args.automatic_transform_initialization
    verbose = args.verbose

    from squirrel.workflows.elastix import slices_to_volume

    slices_to_volume(
        moving_filepath,
        fixed_filepath,
        out_filepath,
        moving_key=moving_key,
        fixed_key=fixed_key,
        transform=transform,
        automatic_transform_initialization=automatic_transform_initialization,
        verbose=verbose
    )


def amst():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs Alignment to Median Smoothed Template (AMST)',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('pre_aligned_stack', type=str,
                        help='Input filepath for an aligned image stack (h5, ome.zarr or tif stack)')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the transformations (*.json)')
    parser.add_argument('--raw_stack', type=str, default=None,
                        help='Input of the raw stack from which the pre-alignment was derived '
                             '(Note: Not yet implemented)')
    parser.add_argument('--pre_align_key', type=str, default='data',
                        help='Internal path of the input; default="data"; used if stack is h5 or ome.zarr')
    parser.add_argument('--pre_align_pattern', type=str, default='*.tif',
                        help='Used to glob tif files from a tif stack folder; default="*.tif"')
    parser.add_argument('--transform', type=str, default='affine',
                        help='The transformation to use for alignment to the template; default="affine"')
    parser.add_argument('--auto_mask_off', action='store_true',
                        help='Turn off automatic generation of a mask for fixed and moving image')
    # parser.add_argument('--number_of_spatial_samples', type=int, default=None,
    #                     help='Elastix parameter')
    # parser.add_argument('--maximum_number_of_iterations', type=int, default=None,
    #                     help='Elastix parameter')
    # parser.add_argument('--number_of_resolutions', type=int, default=None,
    #                     help='Elastix parameter')
    # parser.add_argument('--pre_fix_big_jumps', action='store_true',
    #                     help='Determines big jumps and fixes them using cross-correlation')
    parser.add_argument('--median_radius', type=int, default=7,
                        help='Radius of the z-median-smoothing used to create the template stack; default=7')
    parser.add_argument('--gaussian_sigma', type=float, default=0.,
                        help='Pre-smooth the images using a gaussian filter; default=0. (no smoothing)')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
    parser.add_argument('--elastix_parameter_filepath', type=str, default=None,
                        help='Filepath of an elastix parameter file; default=None')
    parser.add_argument('--crop_to_bounds_off', action='store_true',
                        help='Switches off automated cropping to image bounds')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    pre_aligned_stack = args.pre_aligned_stack
    out_filepath = args.out_filepath
    raw_stack = args.raw_stack
    pre_align_key = args.pre_align_key
    pre_align_pattern = args.pre_align_pattern
    transform = args.transform
    auto_mask_off = args.auto_mask_off
    median_radius = args.median_radius
    gaussian_sigma = args.gaussian_sigma
    z_range = args.z_range
    elastix_parameter_filepath = args.elastix_parameter_filepath
    crop_to_bounds_off = args.crop_to_bounds_off
    verbose = args.verbose

    from squirrel.workflows.amst import amst_workflow

    amst_workflow(
        pre_aligned_stack,
        out_filepath,
        raw_stack=raw_stack,
        pre_align_key=pre_align_key,
        pre_align_pattern=pre_align_pattern,
        transform=transform,
        auto_mask_off=auto_mask_off,
        median_radius=median_radius,
        gaussian_sigma=gaussian_sigma,
        z_range=z_range,
        elastix_parameters=elastix_parameter_filepath,
        crop_to_bounds_off=crop_to_bounds_off,
        verbose=verbose
    )


def elastix_stack_alignment():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs a stack alignment with Elastix',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack', type=str,
                        help='Input filepath for the image stack (h5 or tif stack)')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the transformations (*.json)')
    parser.add_argument('--key', type=str, default='data',
                        help='Internal path of the input; default="data"; used if stack is h5 file')
    parser.add_argument('--transform', type=str, default='translation',
                        help='The transformation; default="translation"')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='Used to glob tif files from a tif stack folder; default="*.tif"')
    parser.add_argument('--auto_mask', action='store_true',
                        help='Automatically generates a mask for fixed and moving image')
    parser.add_argument('--number_of_spatial_samples', type=int, default=None,
                        help='Elastix parameter')
    parser.add_argument('--maximum_number_of_iterations', type=int, default=None,
                        help='Elastix parameter')
    parser.add_argument('--number_of_resolutions', type=int, default=None,
                        help='Elastix parameter')
    # parser.add_argument('--extended_output', action='store_true',
    #                     help='Increase information content of the output')
    parser.add_argument('--pre_fix_big_jumps', action='store_true',
                        help='Determines big jumps and fixes them using cross-correlation')
    parser.add_argument('--pre_fix_iou_thresh', type=float, default=0.5,
                        help='If the intersection over union of non-background areas of two adjacent sliced exceeds'
                             'this treshold, the big jump prefix is computed')
    parser.add_argument('--gaussian_sigma', type=float, default=0.,
                        help='Perform a gaussian filter before registration')
    parser.add_argument('--parameter_map', type=str, default=None,
                        help='Elastix parameter map file')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
    parser.add_argument('--z_step', type=int, default=1,
                        help='Performs an alignment with every n-th slice only.')
    parser.add_argument('--determine_bounds', action='store_true',
                        help='Appends the bounding box of data within each slice to the results metadata. '
                             'Useful for auto-padding later on')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack = args.stack
    out_filepath = args.out_filepath
    key = args.key
    transform = args.transform
    pattern = args.pattern
    auto_mask = args.auto_mask
    number_of_spatial_samples = args.number_of_spatial_samples
    maximum_number_of_iterations = args.maximum_number_of_iterations
    number_of_resolutions = args.number_of_resolutions
    pre_fix_big_jumps = args.pre_fix_big_jumps
    pre_fix_iou_thresh = args.pre_fix_iou_thresh
    gaussian_sigma = args.gaussian_sigma
    parameter_map = args.parameter_map
    z_range = args.z_range
    z_step = args.z_step
    determine_bounds = args.determine_bounds
    verbose = args.verbose

    from squirrel.workflows.elastix import elastix_stack_alignment_workflow

    elastix_stack_alignment_workflow(
        stack,
        out_filepath,
        transform=transform,
        key=key,
        pattern=pattern,
        auto_mask=auto_mask,
        number_of_spatial_samples=number_of_spatial_samples,
        maximum_number_of_iterations=maximum_number_of_iterations,
        number_of_resolutions=number_of_resolutions,
        pre_fix_big_jumps=pre_fix_big_jumps,
        pre_fix_iou_thresh=pre_fix_iou_thresh,
        gaussian_sigma=gaussian_sigma,
        parameter_map=parameter_map,
        z_range=z_range,
        z_step=z_step,
        determine_bounds=determine_bounds,
        verbose=verbose
    )


def stack_alignment_validation():
    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Validation of a stack alignment',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack', type=str,
                        help='Input filepath for the image stack (h5 or tif stack)')
    parser.add_argument('out_dirpath', type=str,
                        help='Output directory path for the result plots')
    parser.add_argument('rois', nargs='+', type=str,
                        help='Define one or multiple regions for validation in pixels\n'
                             'Format: "z,y,x,d,h,w"\n'
                             'Use "0" to denote an entire axis, e.g.: "0,100,200,0,64,64" will grab a 64^2 region '
                             '  along the entire z-axis')
    parser.add_argument('--key', type=str, default='data',
                        help='Internal path of the input; default="data"; used if stack is h5 file')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='Used to glob tif files from a tif stack folder; default="*.tif"')
    parser.add_argument('--resolution_yx', type=float, nargs=2, default=(1.0, 1.0),
                        help='xy-resolution used to properly plot the results. Default=(1.0, 1.0)')
    parser.add_argument('--out_name', type=str, default=None,
                        help='An additional name for the result files which is added to the filenames')
    parser.add_argument('--y_max', type=int, default=None,
                        help='Maximum of the y-axis')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack = args.stack
    out_dirpath = args.out_dirpath
    rois = args.rois
    key = args.key
    pattern = args.pattern
    resolution_yx = args.resolution_yx
    out_name = args.out_name
    y_max = args.y_max
    verbose = args.verbose

    from squirrel.workflows.elastix import stack_alignment_validation_workflow

    from squirrel.library.string_conversion import str2values
    from squirrel.library.rois import list2roi
    rois = [list2roi(str2values(roi, dtype='int')) for roi in rois]

    stack_alignment_validation_workflow(
        stack,
        out_dirpath,
        rois,
        key=key,
        pattern=pattern,
        resolution_yx=resolution_yx,
        out_name=out_name,
        y_max=y_max,
        verbose=verbose,
    )


def make_elastix_default_parameter_file():
    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Generates a default parameter file for Elastix registration',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('out_filepath', type=str,
                        help='Output directory path for the result plots')
    parser.add_argument('--transform', type=str, default='translation',
                        help='One of the available transforms: ["translation", "affine", "bspline"]; '
                             'default="translation"')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    out_filepath = args.out_filepath
    transform = args.transform
    verbose = args.verbose

    if verbose:
        print(f'out_filepath = {out_filepath}')
        print(f'transform = {transform}')

    from SimpleITK import GetDefaultParameterMap, WriteParameterFile
    params = GetDefaultParameterMap(transform)
    WriteParameterFile(params, out_filepath)


def apply_multi_step_stack_alignment():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Applies an affine transformation on a volume',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('image_stack', type=str,
                        help='Input filepath for the image stack (h5 or tif stack)')
    parser.add_argument('transform_paths', type=str, nargs='+',
                        help='Json file or folders containing the transformations for each slice')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the result file (only h5 for now)')
    parser.add_argument('--key', type=str, default='data',
                        help='Internal path of the input; default="data"; used if stack is h5 file')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='Used to glob tif files from a tif stack folder; default="*.tif"')
    parser.add_argument('--auto_pad', action='store_true',
                        help='Automatically adjust the canvas size of the output stack to best fit the data')
    parser.add_argument('--target_image_shape', type=int, nargs=3, default=None,
                        help='Pre-define a stack shape for the output stack; default=None')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='The number of cores to use for processing')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    image_stack = args.image_stack
    transform_paths = args.transform_paths
    out_filepath = args.out_filepath
    key = args.key
    pattern = args.pattern
    auto_pad = args.auto_pad
    target_image_shape = args.target_image_shape
    z_range = args.z_range
    n_workers = args.n_workers
    verbose = args.verbose

    from squirrel.workflows.elastix import apply_multi_step_stack_alignment_workflow

    apply_multi_step_stack_alignment_workflow(
        image_stack,
        transform_paths,
        out_filepath,
        key=key,
        pattern=pattern,
        auto_pad=auto_pad,
        target_image_shape=target_image_shape,
        z_range=z_range,
        write_result=True,
        n_workers=n_workers,
        verbose=verbose
    )


if __name__ == '__main__':

    # elastix_stack_alignment()
    # amst()
    # stack_alignment_validation()
    apply_multi_step_stack_alignment()
