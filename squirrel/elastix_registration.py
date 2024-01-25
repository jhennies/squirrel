
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
        automatic_transform_initialization=automatic_transform_initialization,
        view_results_in_napari=view_results_in_napari,
        verbose=verbose
    )


def affine3d():

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
    parser.add_argument('out_path', type=str,
                        help='Output folder for all results files (must be empty or not existing)')
    parser.add_argument('--moving_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--fixed_key', type=str, default='data',
                        help='Internal path of the fixed input; default="data"; used if fixed_filepath is h5 file')
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
    automatic_transform_initialization = args.automatic_transform_initialization
    view_results_in_napari = args.view_results_in_napari
    verbose = args.verbose

    from squirrel.workflows.elastix import affine3d

    affine3d(
        moving_filepath,
        fixed_filepath,
        out_path,
        moving_key=moving_key,
        fixed_key=fixed_key,
        automatic_transform_initialization=automatic_transform_initialization,
        view_results_in_napari=view_results_in_napari,
        verbose=verbose
    )


if __name__ == '__main__':
    affine3d()
