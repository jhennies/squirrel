
def morphological_operation():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Performs a morphological operation on a label volume',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack_path', type=str,
                        help='Paths of the input volume')
    parser.add_argument('out_path', type=str,
                        help='Output location. Must be either a directory or a file name'
                             'Will be created if not existing')
    parser.add_argument('--key', type=str, default='data',
                        help='For h5 or ome.zarr input stacks this key is used to locate the dataset inside the stack '
                             'location; default="data"')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File pattern to search for within the input folder; default = "*.tif"')
    parser.add_argument('--operation', type=str, default='dilation',
                        help='Morphological operation to perform on the volume; default="dilation"; '
                             'possible values: ("dilation", "erosion", "opening", "closing")')
    # parser.add_argument('--target_dtype', type=str, default=None,
    #                     help='If set, the result will be casted to the respective data type')
    # parser.add_argument('--n_workers', type=int, default=1,
    #                     help='Number of CPUs to use')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack_path = args.stack_path
    out_path = args.out_path
    key = args.key
    pattern = args.pattern
    operation = args.operation
    verbose = args.verbose

    from squirrel.workflows.morphological import morphological_operation_workflow

    morphological_operation_workflow(
        stack_path,
        out_path,
        key=key,
        pattern=pattern,
        operation=operation,
        verbose=verbose
    )
