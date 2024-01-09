def main():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Merge two MoBIE projects',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('mobie_a', type=str,
                        help='Location of first mobie project')
    parser.add_argument('mobie_b', type=str,
                        help='Location of second mobie project')
    parser.add_argument('out_folder', type=str,
                        help='Output folder where the results will be written to')
    parser.add_argument('--datasets_a', type=str, nargs='+', default=None,
                        help='Which datasets to use from the first mobie project, by default all datasets are used')
    parser.add_argument('--datasets_b', type=str, nargs='+', default=None,
                        help='Which datasets to use from the second mobie project, by default all datasets are used')
    parser.add_argument('-copy', '--copy_data', action='store_true',
                        help='Copy the data, by default only xml files are created in the output dataset')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    mobie_a = args.mobie_a
    mobie_b = args.mobie_b
    out_folder = args.out_folder
    datasets_a = args.datasets_a
    datasets_b = args.datasets_b
    copy_data = args.copy_data
    verbose = args.verbose

    from squirrel.mobie import merge_mobie_projects

    merge_mobie_projects(
        mobie_a,
        mobie_b,
        out_folder,
        datasets_a=datasets_a,
        datasets_b=datasets_b,
        copy_data=copy_data,
        verbose=verbose
    )


if __name__ == '__main__':
    main()
