
import os.path


def parse_navigator_file():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Parses a navigator file and writes it out as a dictionary (json format).',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('nav_filepath', type=str,
                        help='Path to navigator file')
    parser.add_argument('-out', '--output_filepath', type=str, default=None,
                        help='Where to save the result (*.json); '
                             'default=None: output will be written to file with the same basename as the input')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    nav_filepath = args.nav_filepath
    output_filepath = args.output_filepath
    verbose = args.verbose

    from squirrel.workflows.serial_em import parse_navigator_file_workflow

    # If the workflow is called as an API call with output_filepath=None the result will not be written to file and just
    #    be returned by the function
    # If this is directly called from console there has to be an output filepath:
    if output_filepath is None:
        output_filepath = f'{os.path.splitext(nav_filepath)[0]}.json'

    parse_navigator_file_workflow(
        nav_filepath,
        output_filepath=output_filepath,
        verbose=verbose,
    )


def create_link_maps():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Draws locations of view and search locations on search and grid maps, respectively',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('nav_filepath', type=str,
                        help='Path to navigator file')
    parser.add_argument('out_dirpath', type=str, default=None,
                        help='Path to where the results will be saved')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    nav_filepath = args.nav_filepath
    out_dirpath = args.out_dirpath
    verbose = args.verbose

    from squirrel.workflows.serial_em import create_link_maps_workflow

    create_link_maps_workflow(
        nav_filepath,
        out_dirpath,
        verbose=verbose,
    )
