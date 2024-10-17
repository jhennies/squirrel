import os.path


def get_mobie_table_path(project_path):
    return os.path.join(project_path, 'mobie.csv')


def append_mobie_table(table_filepath, entry):

    import pandas as pd

    table_data = pd.DataFrame()
    if os.path.exists(table_filepath):
        table_data = pd.read_csv(table_filepath, sep='\t')

    new_table_data = pd.concat([table_data, pd.DataFrame(entry)], ignore_index=True, sort=False)
    new_table_data = new_table_data.fillna('')

    new_table_data.to_csv(table_filepath, index=False, sep='\t')


def remove_mobie_table_entry(table_filepath, value, row='uri', verbose=False, debug=False):

    import pandas as pd

    table_data = pd.read_csv(table_filepath, sep='\t')
    if verbose:
        print(table_data)

    table_data = table_data[table_data[row] != value]
    table_data = table_data.fillna('')

    if verbose:
        print(table_data)

    if not debug:
        table_data.to_csv(table_filepath, index=False, sep='\t')


def replace_mobie_table(table_filepath, entries):

    import pandas as pd

    table_data = pd.DataFrame(entries)
    table_data.to_csv(table_filepath, index=False, sep='\t')


def update_mobie_table_entry(table_filepath, entry, item):
    assert len(entry) == 2, 'entry should be a list with two items: a column header and a value'
    assert len(item) == 2, 'item should be a list with two items: a column header and a value to look for'

    import pandas as pd
    mobie_table = pd.read_csv(table_filepath, sep='\t')

    row_index = mobie_table[mobie_table[item[0]] == item[1]].index[0]
    mobie_table.at[row_index, entry[0]] = entry[1]

    mobie_table.to_csv(table_filepath, index=False, sep='\t')


def init_mobie_table(table_filepath, data_map_filepaths, verbose=False):
    if verbose:
        print(f'table_filepath = {table_filepath}')
        print(f'data_map_filepaths = {data_map_filepaths}')


class MobieTableProject():

    def __init__(self, project_dirpath, verbose=False):

        self._project_dirpath = project_dirpath
        self._verbose = verbose

        if not os.path.exists(project_dirpath):
            os.mkdir(project_dirpath)
            return

    def init_with_data_maps(self, data_map_filepaths):
        assert len(os.listdir(self._project_dirpath)) == 0, 'Can only initialize a project in an empty directory'

        self._mobie_table_path = get_mobie_table_path(self._project_dirpath)
        init_mobie_table(self._mobie_table_path, data_map_filepaths, verbose=self._verbose)

    def get_project_dirpath(self):
        return self._project_dirpath



