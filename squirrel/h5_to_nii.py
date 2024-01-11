
def main():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert a dataset within a h5 container to a nifti volume',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('h5_file', type=str,
                        help='Input h5 container')
    parser.add_argument('h5_key', type=str,
                        help='Internal path of the dataset')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath where the results will be written to')
    parser.add_argument('-ax', '--axes_order', type=str, default='zyx',
                        help='Re-define the order of the volume axes')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    h5_file = args.h5_file
    h5_key = args.h5_key
    out_filepath = args.out_filepath
    axes_order = args.axes_order
    verbose = args.verbose

    from squirrel.workflows.convert import h5_to_nii

    h5_to_nii(
        h5_file,
        h5_key,
        out_filepath,
        axes_order=axes_order,
        verbose=verbose
    )


if __name__ == '__main__':
    main()
