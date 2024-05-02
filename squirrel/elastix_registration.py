
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
                        help='Radius of the z-median-smoothing used to create the template stack')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
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
    z_range = args.z_range
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
        z_range=z_range,
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
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
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
    z_range = args.z_range
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
        z_range=z_range,
        verbose=verbose
    )


if __name__ == '__main__':

    # elastix_stack_alignment()
    amst()
