
def main():

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

    from squirrel.workflows.normalization import normalize_slices

    normalize_slices(
        in_path,
        out_path,
        pattern=pattern,
        in_h5_key=in_h5_key,
        out_h5_key=out_h5_key,
        verbose=verbose
    )


if __name__ == '__main__':
    main()
