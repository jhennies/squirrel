
def init_mobie_project(project_dirpath, data_map_filepaths, verbose=False):

    from squirrel.library.table_mobie import MobieTableProject

    mtp = MobieTableProject(project_dirpath, verbose=verbose)
    mtp.init_with_data_maps(data_map_filepaths)
