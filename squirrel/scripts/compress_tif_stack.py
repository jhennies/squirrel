
def main():

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
                        help='File patter to search for within the input folder; default = "*.tif"')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    in_folder = args.in_folder
    out_folder = args.out_folder
    pattern = args.pattern
    verbose = args.verbose

    from squirrel.convert import compress_tif_stack

    compress_tif_stack(
        in_folder,
        out_folder,
        pattern=pattern,
        verbose=verbose
    )


if __name__ == '__main__':
    main()
