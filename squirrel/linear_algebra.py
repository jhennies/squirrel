
def dot_product_on_affines():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs the dot product on two affine transformations '
                    'or element-wise on two sequences of affine transformations.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('transform_filepaths', type=str, nargs=2,
                        help='Json files containing the affine transformation(s)')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the result file (*.json)')
    parser.add_argument('--inverse', type=int, nargs=2, default=(0, 0),
                        help='Defines whether the inverse of an input is used; Default=(0, 0)')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    transform_filepaths = args.transform_filepaths
    out_filepath = args.out_filepath
    inverse = args.inverse
    verbose = args.verbose

    from squirrel.workflows.transformation import dot_product_on_affines_workflow
    dot_product_on_affines_workflow(
        transform_filepaths,
        out_filepath,
        inverse=inverse,
        verbose=verbose,
    )


def scale_sequential_affines():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Element-wise scaling of a sequence of affine transformations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the affine transformations')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the result file (*.json)')
    parser.add_argument('scale', type=float,
                        help='The scaling factor to be applied')
    parser.add_argument('--xy_pivot', nargs=2, type=float, default=(0., 0.),
                        help='A pivot point of the 2D affine transformations')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    transform_filepath = args.transform_filepath
    out_filepath = args.out_filepath
    scale = args.scale
    xy_pivot = args.xy_pivot
    verbose = args.verbose

    from squirrel.workflows.transformation import scale_sequential_affines_workflow
    scale_sequential_affines_workflow(
        transform_filepath,
        out_filepath,
        scale,
        xy_pivot=xy_pivot,
        verbose=verbose,
    )


def apply_affine_sequence():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Each element of an affine transform sequence is applied to the result of the previous',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the affine transformations')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the result file (*.json)')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    transform_filepath = args.transform_filepath
    out_filepath = args.out_filepath
    verbose = args.verbose

    from squirrel.workflows.transformation import serialize_affine_sequence_workflow
    serialize_affine_sequence_workflow(
        transform_filepath,
        out_filepath,
        verbose=verbose,
    )


def smooth_affine_sequence():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Smoothing of an affine sequence.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the affine transformations')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the result file (*.json)')
    parser.add_argument('sigma', type=float,
                        help='Sigma of the Gaussian kernel')
    parser.add_argument('--components', type=str, nargs='+', default=None,
                        help='Which components of the affine transform to smooth; Defaults to all of them\n'
                             'Possible values: ["translation", "rotation", "shear", "scale"]\n'
                             'E.g., "--components translation rotation" smoothes translation and rotation while '
                             'leaving shearing and scaling as it is.')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    transform_filepath = args.transform_filepath
    out_filepath = args.out_filepath
    sigma = args.sigma
    components = args.components
    verbose = args.verbose

    from squirrel.workflows.transformation import smooth_affine_sequence_workflow
    smooth_affine_sequence_workflow(
        transform_filepath,
        out_filepath,
        sigma,
        components=components,
        verbose=verbose,
    )


def inverse_of_sequence():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Computing the inverse matrices for each element of a sequence',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the affine transformations')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the result file (*.json)')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    transform_filepath = args.transform_filepath
    out_filepath = args.out_filepath
    verbose = args.verbose

    from squirrel.workflows.transformation import inverse_of_sequence_workflow
    inverse_of_sequence_workflow(
        transform_filepath,
        out_filepath,
        verbose=verbose,
    )


def add_translational_drift():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Adds translational drift to an affine sequence',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the affine transformations')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the result file (*.json)')
    parser.add_argument('--drift', type=float, nargs=2, metavar=('Y', 'X'), default=(0., 0.),
                        help='The x- and y- drift that is added to each transformation in pixels')
    parser.add_argument('--is_serialized', action='store_true',
                        help='Add this flag if the sequence is serialized. Then the drift will also be serialized '
                             'before adding to each element of the sequence.')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    transform_filepath = args.transform_filepath
    out_filepath = args.out_filepath
    drift = args.drift
    is_serialized = args.is_serialized
    verbose = args.verbose

    from squirrel.workflows.transformation import add_translational_drift_workflow
    add_translational_drift_workflow(
        transform_filepath,
        out_filepath,
        drift,
        is_serialized=is_serialized,
        verbose=verbose
    )


def modify_step_in_sequence():
    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Modify one element of a sequence of affine transformations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('transform_filepath', type=str,
                        help='Json file containing the transformation')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath for the result file (*.json)')
    parser.add_argument('idx', type=int,
                        help='The index of the step which will be modified')
    parser.add_argument('--affine', type=float, nargs=6, default=(1., 0., 0., 0., 1., 0.),
                        help='Affine transform which will be applied to the specified step')
    parser.add_argument('--replace', action='store_true',
                        help='If set, the transform at the specified step will be replaced. '
                             'np.dot will be used otherwise')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    transform_filepath = args.transform_filepath
    out_filepath = args.out_filepath
    idx = args.idx
    affine = args.affine
    replace = args.replace
    verbose = args.verbose

    from squirrel.workflows.transformation import modify_step_in_sequence_workflow
    modify_step_in_sequence_workflow(
        transform_filepath,
        out_filepath,
        idx,
        affine,
        replace=replace,
        verbose=verbose
    )


def create_affine_sequence():
    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Create an affine sequence',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('out_filepath', type=str,
                        help='Where the result will be saved')
    parser.add_argument('length', type=int,
                        help='The length of the sequence')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    out_filepath = args.out_filepath
    length = args.length
    verbose = args.verbose

    from squirrel.workflows.transformation import create_affine_sequence_workflow
    create_affine_sequence_workflow(out_filepath, length, verbose=verbose)


if __name__ == '__main__':
    add_translational_drift()
