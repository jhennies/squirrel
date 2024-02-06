
def main():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert a Microscopy Image Browser model to a tif stack',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('mib_model_file', type=str,
                        help='Input MIB model file')
    parser.add_argument('out_folder', type=str,
                        help='Output folder where the results will be written to')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    mib_model_file = args.mib_model_file
    out_folder = args.out_folder
    verbose = args.verbose

    from squirrel.workflows.convert import mib_to_tif

    mib_to_tif(
        mib_model_file,
        out_folder,
        verbose=verbose
    )


if __name__ == '__main__':
    main()
