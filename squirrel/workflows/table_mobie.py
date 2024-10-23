
def init_mobie_project_workflow(
        project_dirpath,
        data_map_filepaths,
        types='intensities',
        views='raw',
        groups='group0',
        verbose=False
):

    from squirrel.library.table_mobie import MobieTableProject

    mtp = MobieTableProject(project_dirpath, verbose=verbose)
    mtp.init_with_data_maps(
        data_map_filepaths,
        types=types,
        views=views,
        groups=groups
    )
