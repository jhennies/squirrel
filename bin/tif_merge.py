
if __name__ == '__main__':

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

    from squirrel.convert import merge_tif_stacks

    merge_tif_stacks(
        stack_folders,
        out_folder,
        pattern=pattern,
        out_pattern=out_pattern,
        pad_canvas=pad_canvas,
        verbose=verbose
    )
