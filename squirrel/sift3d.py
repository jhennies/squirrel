"""
This wraps the 3D SIFT algorithm from https://github.com/bbrister/SIFT3D
Consequently SIFT3D needs to be installed on the system with the bin directory present in PATH variable
"""


def main():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs the 3D SIFT algorithm implemented in https://github.com/bbrister/SIFT3D',
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
    parser.add_argument('--nn_thresh', type=float, default=0.8,
                        help='See regSIFT3D doc')
    parser.add_argument('--corner_thresh', type=float, default=0.4,
                        help='See regSIFT3D doc')
    parser.add_argument('--num_kp_levels', type=float, default=3,
                        help='See regSIFT3D doc')
    parser.add_argument('-napari', '--view_results_in_napari', action='store_true',
                        help='Show the results in the napari viewer')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    moving_filepath = args.moving_filepath
    fixed_filepath = args.fixed_filepath
    out_path = args.out_path
    moving_key = args.moving_key
    fixed_key = args.fixed_key
    nn_thresh = args.nn_thresh
    corner_thresh = args.corner_thresh
    num_kp_levels = args.num_kp_levels
    view_results_in_napari = args.view_results_in_napari
    verbose = args.verbose

    from squirrel.workflows.sift3d import sift3d_workflow

    sift3d_workflow(
        moving_filepath,
        fixed_filepath,
        out_path,
        moving_key=moving_key,
        fixed_key=fixed_key,
        nn_thresh=nn_thresh,
        corner_thresh=corner_thresh,
        num_kp_levels=num_kp_levels,
        view_results_in_napari=view_results_in_napari,
        verbose=verbose
    )


if __name__ == '__main__':
    main()
