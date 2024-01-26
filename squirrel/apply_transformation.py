
def affine_on_volume():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Applies an affine transformation on a volume',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('image_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the transformation')
    parser.add_argument('-o', '--out_filepath', type=str, default=None,
                        help='Output filepath for the result file (only h5 for now)')
    parser.add_argument('--image_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--no_offset_to_center', action='store_true',
                        help="If set, the image is rotated around it's origin")
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    image_filepath = args.image_filepath
    transform_filepath = args.transform_filepath
    out_filepath = args.out_filepath
    image_key = args.image_key
    no_offset_to_center = args.no_offset_to_center
    verbose = args.verbose

    from squirrel.workflows.transformation import apply_affine
    apply_affine(
        image_filepath,
        transform_filepath,
        out_filepath=out_filepath,
        image_key=image_key,
        no_offset_to_center=no_offset_to_center,
        verbose=verbose
    )


def sequential_affine_on_volume():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Applies an affine transformation on a volume',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('image_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('transform_filepaths', type=str, nargs='+',
                        help='Json or csv file(s) containing the transformations')
    parser.add_argument('-o', '--out_filepath', type=str, default=None,
                        help='Output filepath for the result file (only h5 for now)')
    parser.add_argument('--image_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--no_offset_to_center', action='store_true',
                        help="If set, the image is rotated around it's origin")
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    image_filepath = args.image_filepath
    transform_filepaths = args.transform_filepaths
    out_filepath = args.out_filepath
    image_key = args.image_key
    no_offset_to_center = args.no_offset_to_center
    verbose = args.verbose

    from squirrel.workflows.transformation import apply_sequential_affine
    apply_sequential_affine(
        image_filepath,
        transform_filepaths,
        out_filepath=out_filepath,
        image_key=image_key,
        no_offset_to_center=no_offset_to_center,
        verbose=verbose
    )


def average_affine_on_volume():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Applies an average of multiple affine transforms to a volume.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('image_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the transformation')
    parser.add_argument('-o', '--out_filepath', type=str, default=None,
                        help='Output filepath for the result file (only h5 for now)')
    parser.add_argument('--image_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    image_filepath = args.image_filepath
    transform_filepath = args.transform_filepath
    out_filepath = args.out_filepath
    image_key = args.image_key
    verbose = args.verbose

    from squirrel.workflows.transformation import apply_average_affine
    apply_average_affine(
        image_filepath,
        transform_filepath,
        out_filepath=out_filepath,
        image_key=image_key,
        verbose=verbose
    )


def apply_rotation_and_scale():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Applies an average of multiple affine transforms to a volume.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('image_filepath', type=str,
                        help='Input filepath for the moving volume (nii or h5)')
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the transformation')
    parser.add_argument('stack_item_distance', type=int,
                        help='The distance in pixels of the sub-stacks')
    parser.add_argument('-o', '--out_filepath', type=str, default=None,
                        help='Output filepath for the result file (only h5 for now)')
    parser.add_argument('--image_key', type=str, default='data',
                        help='Internal path of the moving input; default="data"; used if moving_filepath is h5 file')
    parser.add_argument('--apply', type=str, nargs='+', default=('rotates', 'scale', 'translate'),
                        help='Applies the defined transform components to the volume')
    parser.add_argument('--pivot', type=float, nargs=3, default=(0., 0., 0.),
                        help='Point around which scaling is centered. Caution: This does not work for rotation (yet?)')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    image_filepath = args.image_filepath
    transform_filepath = args.transform_filepath
    stack_item_distance = args.stack_item_distance
    out_filepath = args.out_filepath
    image_key = args.image_key
    apply = args.apply
    pivot = args.pivot
    verbose = args.verbose

    from squirrel.workflows.transformation import apply_rotation_and_scale_from_transform_stack
    apply_rotation_and_scale_from_transform_stack(
        image_filepath,
        transform_filepath,
        stack_item_distance,
        out_filepath=out_filepath,
        image_key=image_key,
        apply=apply,
        pivot=pivot,
        verbose=verbose
    )


if __name__ == '__main__':
    affine_on_volume()
