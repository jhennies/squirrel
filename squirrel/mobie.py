
def init_mobie_project():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Initialize a MoBIE table project with one or multiple volume data maps (assuming *.ome.zarr)',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('project_dirpath', type=str,
                        help='The path in which the project will be created')
    parser.add_argument('data_map_filepaths', nargs='+', type=str,
                        help='One or multiple paths of *.ome.zarr volume data maps')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    project_dirpath = args.project_dirpath
    data_map_filepaths = args.data_map_filepaths
    verbose = args.verbose

    from squirrel.workflows.table_mobie import init_mobie_project

    init_mobie_project(project_dirpath, data_map_filepaths, verbose=verbose)
