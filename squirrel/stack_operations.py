
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
    parser.add_argument('--in_pattern', type=str, default='*.tif',
                        help='File patter to search for within the input folder; default = "*.tif"; used if in_path is '
                             'tif stack')
    parser.add_argument('--in_key', type=str, default='data',
                        help='Internal path of the input dataset; default="data"; used if in_path is h5 filepath')
    parser.add_argument('--out_key', type=str, default='data',
                        help='Internal path of the output dataset; default="data"; used if out_path is h5 filepath')
    parser.add_argument('--dilate_background', type=int, default=0,
                        help='Dilate the background before computing data quantiles. default=0 (off)')
    parser.add_argument('--quantiles', type=float, nargs=2, default=(0.1, 0.9),
                        help='Lower and upper quantile of the gray value spectrum that is used to normalize;'
                             'default=(0.1, 0.9)')
    parser.add_argument('--anchors', type=float, nargs=2, default=(0.2, 0.8),
                        help='The lower and upper quantiles of the input gray value spectrum are transferred to these '
                             'relative values; default=(0.2, 0.8)')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='The number of cores to use for processing')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path
    in_pattern = args.in_pattern
    in_key = args.in_key
    out_key = args.out_key
    dilate_background = args.dilate_background
    quantiles = args.quantiles
    anchors = args.anchors
    n_workers = args.n_workers
    verbose = args.verbose

    from squirrel.workflows.normalization import normalize_slices_workflow

    normalize_slices_workflow(
        in_path,
        out_path,
        in_pattern=in_pattern,
        in_key=in_key,
        out_key=out_key,
        dilate_background=dilate_background,
        quantiles=quantiles,
        anchors=anchors,
        n_workers=n_workers,
        verbose=verbose
    )


def merge_tif_stacks():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Merge two tif stacks',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack_folders', nargs='+', type=str,
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
    parser.add_argument('--inconsistent_shapes', action='store_true',
                        help='Enable this if tif slices within one stack may have different shapes;'
                             'This setting massively increases computation time!')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack_folders = args.stack_folders
    out_folder = args.out_folder
    pattern = args.pattern
    out_pattern = args.out_pattern
    pad_canvas = args.pad_canvas
    inconsistent_shapes = args.inconsistent_shapes
    verbose = args.verbose

    from squirrel.workflows.convert import merge_tif_stacks_workflow

    merge_tif_stacks_workflow(
        stack_folders,
        out_folder,
        pattern=pattern,
        out_pattern=out_pattern,
        pad_canvas=pad_canvas,
        inconsistent_shapes=inconsistent_shapes,
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


def stack_calculator():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Performs mathematical operations on two image stacks. E.g. pairwise stack_a + stack_b',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack_paths', type=str, nargs=2,
                        metavar=('A', 'B'),
                        help='Path of the input stacks')
    parser.add_argument('out_path', type=str,
                        help='Output location. Must be either a directory or an h5 file name. '
                             'Will be created if not existing')
    parser.add_argument('--keys', type=str, default=('data', 'data'), nargs=2,
                        metavar=('A', 'B'),
                        help='For h5 or ome.zarr input stacks this key is used to locate the dataset inside the stack '
                             'location; default=("data", "data"); '
                             'Input an empty string if not applicable for one of the stacks, '
                             'e.g. `--keys "" "data_b"` or `--keys "data_a" ""`')
    parser.add_argument('--patterns', type=str, default=('*.tif', '*.tif'),
                        metavar=('A', 'B'),
                        help='File pattern to search for within the input folder; default = ("*.tif", "*.tif"); '
                             'Input an empty string if not applicable for one of the stacks (see --keys)')
    parser.add_argument('--operation', type=str, default='add',
                        help='Mathematical operation to perform on image slice pairs; default="add"; '
                             'possible values: ("add", "subtract", "multiply", "divide", "min", "max", "average"')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack_paths = args.stack_paths
    out_path = args.out_path
    keys = args.keys
    patterns = args.patterns
    operation = args.operation
    verbose = args.verbose

    from squirrel.workflows.volume import stack_calculator_workflow

    stack_calculator_workflow(
        stack_paths,
        out_path,
        keys=keys,
        patterns=patterns,
        operation=operation,
        verbose=verbose
    )
