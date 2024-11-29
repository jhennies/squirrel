
def sift_log_to_affine_stack():
    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description="Converts Fiji's Linear Stack Alignment with SIFT log output to an affine stack file",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('log_filepath', type=str,
                        help='Json file containing the SIFT log output')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the result file (*.json)')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    log_filepath = args.log_filepath
    out_filepath = args.out_filepath
    verbose = args.verbose

    from squirrel.workflows.fiji import sift_log_to_affine_stack_workflow

    sift_log_to_affine_stack_workflow(log_filepath, out_filepath=out_filepath, verbose=verbose)

