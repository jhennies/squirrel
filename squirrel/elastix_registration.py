
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
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack = args.stack
    out_filepath = args.out_filepath
    key = args.key
    transform = args.transform
    pattern = args.pattern
    verbose = args.verbose

    from squirrel.workflows.elastix import elastix_stack_alignment_workflow

    elastix_stack_alignment_workflow(
        stack,
        out_filepath,
        transform=transform,
        key=key,
        pattern=pattern,
        verbose=verbose
    )


if __name__ == '__main__':
    elastix_slices_to_volume()
