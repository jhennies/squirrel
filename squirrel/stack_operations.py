
def normalize_slices():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Normalizes slices within a tif stack or h5 dataset',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('in_path', type=str,
                        help='Input folder containing a tif stack or filepath of h5 container')
    parser.add_argument('out_path', type=str,
                        help='Output folder for tif stack or filepath for h5 container where the results will '
                             'be written to')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File patter to search for within the input folder; default = "*.tif"; used if in_path is '
                             'tif stack')
    parser.add_argument('--in_h5_key', type=str, default='data',
                        help='Internal path of the input dataset; default="data"; used if in_path is h5 filepath')
    parser.add_argument('--out_h5_key', type=str, default='data',
                        help='Internal path of the output dataset; default="data"; used if out_path is h5 filepath')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path
    pattern = args.pattern
    in_h5_key = args.in_h5_key
    out_h5_key = args.out_h5_key
    verbose = args.verbose

    from squirrel.workflows.normalization import normalize_slices_workflow

    normalize_slices_workflow(
        in_path,
        out_path,
        pattern=pattern,
        in_h5_key=in_h5_key,
        out_h5_key=out_h5_key,
        verbose=verbose
    )


def merge_tif_stacks():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Merge two tif stacks',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack_folders', nargs=2, type=str,
                        help='List of input folders')
    parser.add_argument('out_folder', type=str,
                        help='Output folder where the results will be written to')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File pattern for globbing the input stack; default="*.tif"')
    parser.add_argument('--out_pattern', type=str, default='slice_{:05d}.tif',
                        help='File format string defining the output filenames; default="slice_{:05d}.tif')
    parser.add_argument('--pad_canvas', action='store_true',
                        help='For non-equal canvas sizes this can be used to generate a common canvas size for the '
                             'entire result stack')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack_folders = args.stack_folders
    out_folder = args.out_folder
    pattern = args.pattern
    out_pattern = args.out_pattern
    pad_canvas = args.pad_canvas
    verbose = args.verbose

    from squirrel.workflows.convert import merge_tif_stacks_workflow

    merge_tif_stacks_workflow(
        stack_folders,
        out_folder,
        pattern=pattern,
        out_pattern=out_pattern,
        pad_canvas=pad_canvas,
        verbose=verbose
    )


def compress_tif_stack():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Create a new compressed copy of a tif stack',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('in_folder', type=str,
                        help='Input folder containing a tif stack')
    parser.add_argument('out_folder', type=str,
                        help='Output folder where the results will be written to')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File pattern to search for within the input folder; default = "*.tif"')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    in_folder = args.in_folder
    out_folder = args.out_folder
    pattern = args.pattern
    verbose = args.verbose

    from squirrel.workflows.convert import compress_tif_stack_workflow

    compress_tif_stack_workflow(
        in_folder,
        out_folder,
        pattern=pattern,
        verbose=verbose
    )


def crop_from_stack():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Crops a region of interest from a stack and saves it as tif or h5',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack_path', type=str,
                        help='Path of the input stack')
    parser.add_argument('out_path', type=str,
                        help='Output location. Must be either a directory or an h5 file name. '
                             'Will be created if not existing')
    parser.add_argument('--roi', type=int, nargs=6, default=None,
                        metavar=('Z', 'Y', 'X', 'D', 'H', 'W'),
                        # metavar=('Z', 'Y', 'X', 'depth', 'height', 'width'),
                        help='Region of interest to crop, given in voxels')
    parser.add_argument('--key', type=str, default='data',
                        help='For h5 or ome.zarr input stacks this key is used to locate the dataset inside the stack '
                             'location')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File pattern to search for within the input folder; default = "*.tif"')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack_path = args.stack_path
    out_path = args.out_path
    roi = args.roi
    key = args.key
    pattern = args.pattern
    verbose = args.verbose

    assert roi is not None

    from squirrel.workflows.volume import crop_from_stack_workflow

    crop_from_stack_workflow(
        stack_path,
        out_path,
        roi,
        key=key,
        pattern=pattern,
        verbose=verbose
    )
