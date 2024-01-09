
def main():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert a dataset within a h5 container to a tif stack',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('h5_file', type=str,
                        help='Input h5 container')
    parser.add_argument('h5_key', type=str,
                        help='Internal path of the dataset')
    parser.add_argument('out_folder', type=str,
                        help='Output folder where the results will be written to')
    parser.add_argument('-ax', '--axes_order', type=str, default='zyx',
                        help='Re-define the order of the volume axes')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    h5_file = args.h5_file
    h5_key = args.h5_key
    out_folder = args.out_folder
    axes_order = args.axes_order
    verbose = args.verbose

    from squirrel.convert import h5_to_tif

    h5_to_tif(
        h5_file,
        h5_key,
        out_folder,
        axes_order=axes_order,
        verbose=verbose
    )


if __name__ == '__main__':
    main()
