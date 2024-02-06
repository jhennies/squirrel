

def sift2d_stack_alignment():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs a stack alignment with SIFT',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack', type=str,
                        help='Input filepath for the image stack (h5 or tif stack)')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the transformations (*.json)')
    parser.add_argument('--key', type=str, default='data',
                        help='Internal path of the input; default="data"; used if stack is h5 file')
    parser.add_argument('--transform', type=str, default='translation',
                        help='The transformation; default="translation"')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='Used to glob tif files from a tif stack folder; default="*.tif"')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack = args.stack
    out_filepath = args.out_filepath
    key = args.key
    transform = args.transform
    pattern = args.pattern
    verbose = args.verbose

    from squirrel.workflows.sift2d import sift_stack_alignment

    sift_stack_alignment(
        stack,
        out_filepath,
        transform=transform,
        key=key,
        pattern=pattern,
        verbose=verbose
    )


if __name__ == '__main__':
    sift2d_stack_alignment()
