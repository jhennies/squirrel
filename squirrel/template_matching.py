
def template_matching_stack_alignment():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs a stack alignment with template matching',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack', type=str,
                        help='Input filepath for the image stack (h5 or tif stack)')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the transformations (*.json)')
    parser.add_argument('--template_roi', type=float, nargs=5, default=None,
                        metavar=('min_x', 'min_y', 'max_x', 'max_y', 'z'),
                        help='Defines where to find the reference template')
    parser.add_argument('--search_roi', type=float, nargs=4, default=None,
                        metavar=('min_x', 'max_x', 'min_y', 'max_y'),
                        help='Where to look for the template; default=None')
    parser.add_argument('--resolution', nargs=3, type=float, default=(1., 1., 1.),
                        metavar=('Z', 'Y', 'X'),
                        help='resolution volume used to compute pixel values for template and search ROIs; '
                             'default=(1., 1., 1.)')
    parser.add_argument('--key', type=str, default='data',
                        help='Internal path of the input; default="data"; used if stack is h5 file')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='Used to glob tif files from a tif stack folder; default="*.tif"')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
    parser.add_argument('--save_template', action='store_true',
                        help='Save the template as tif image')
    parser.add_argument('--determine_bounds', action='store_true',
                        help='Appends the bounding box of data within each slice to the results metadata. '
                             'Useful for auto-padding later on')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack = args.stack
    out_filepath = args.out_filepath
    template_roi = args.template_roi
    search_roi = args.search_roi
    resolution = args.resolution
    key = args.key
    pattern = args.pattern
    z_range = args.z_range
    save_template = args.save_template
    determine_bounds = args.determine_bounds
    verbose = args.verbose

    assert template_roi is not None

    from squirrel.workflows.template_matching import template_matching_stack_alignment_workflow

    template_matching_stack_alignment_workflow(
        stack,
        out_filepath,
        template_roi,
        search_roi=search_roi,
        resolution=resolution,
        key=key,
        pattern=pattern,
        z_range=z_range,
        save_template=save_template,
        determine_bounds=determine_bounds,
        verbose=verbose
    )


if __name__ == '__main__':
    template_matching_stack_alignment()
