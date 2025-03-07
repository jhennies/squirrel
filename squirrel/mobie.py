
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
    parser.add_argument('--types', nargs='+', type=str, default='intensities',
                        help='Map types of the respective data_map_filepaths')
    parser.add_argument('--views', nargs='+', type=str, default='raw',
                        help='View names of the respective data_map_filepaths')
    parser.add_argument('--groups', nargs='+', type=str, default='group0',
                        help='Groups names of the respective data_map_filepaths')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    project_dirpath = args.project_dirpath
    data_map_filepaths = args.data_map_filepaths
    types = args.types
    views = args.views
    groups = args.groups
    verbose = args.verbose

    from squirrel.workflows.table_mobie import init_mobie_project_workflow

    init_mobie_project_workflow(
        project_dirpath,
        data_map_filepaths,
        types=types,
        views=views,
        groups=groups,
        verbose=verbose
    )


def export_rois_with_mobie_table():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Uses information from a MoBIE table to export objects in by their bounding box ROI.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('table_filepath', type=str,
                        help='MoBIE table containing information on labels and their ROIs')
    parser.add_argument('map_dirpath', type=str,
                        help='n5 dirpath of the map from which the data will be extracted')
    parser.add_argument('target_dirpath', type=str,
                        help='n5 dirpath of the map from which the data will be extracted')
    parser.add_argument('-map-key', '--map_key', type=str, default=None,
                        help='Key of the input data map; default=None')
    parser.add_argument('-map-res', '--map_resolution', type=float, nargs=3, default=None,
                        help='Resolution of the input map. Mandatory!')
    parser.add_argument('-mask', '--mask_dirpath', type=str,
                        help='This mask need to have the respective objects as the same ids as specified in the table')
    parser.add_argument('-mask-key', '--mask_key', type=str, default=None,
                        help='Key of the mask map; default=None')
    parser.add_argument('-mask-res', '--mask_resolution', type=float, nargs=3, default=None,
                        help='Resolution of the mask map. Mandatory if mask is given!')
    parser.add_argument('-out-type', '--output_filetype', type=str, default='tif',
                        help='The output data type; default="tif"')
    parser.add_argument('--label_ids', nargs='+', type=int, default=None,
                        help='Which objects defined by their label ID in the table will be extracted; '
                             'default=None denotes all entries')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of CPUs to use')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    table_filepath = args.table_filepath
    map_dirpath = args.map_dirpath
    target_dirpath = args.target_dirpath
    map_key = args.map_key
    map_resolution = args.map_resolution
    mask_dirpath = args.mask_dirpath
    mask_key = args.mask_key
    mask_resolution = args.mask_resolution
    output_filetype = args.output_filetype
    label_ids = args.label_ids
    n_workers = args.n_workers
    verbose = args.verbose

    from squirrel.workflows.mobie import export_rois_with_mobie_table_workflow

    export_rois_with_mobie_table_workflow(
        table_filepath,
        map_dirpath,
        target_dirpath,
        map_key=map_key,
        map_resolution=map_resolution,
        mask_dirpath=mask_dirpath,
        mask_key=mask_key,
        mask_resolution=mask_resolution,
        output_filetype=output_filetype,
        label_ids=label_ids,
        n_workers=n_workers,
        verbose=verbose,
    )


if __name__ == '__main__':

    export_rois_with_mobie_table()
