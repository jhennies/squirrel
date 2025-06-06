
import os


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


def init_mobie_table(
        table_filepath,
        data_map_filepaths,
        name=None,
        types='intensities',
        views='raw',
        groups='group1',
        affine=None,
        blend=None,
        exclusive=None,
        use_abs_path=False,
        verbose=False
):
    if verbose:
        print(f'table_filepath = {table_filepath}')
        print(f'data_map_filepaths = {data_map_filepaths}')

    if os.path.exists(table_filepath):
        os.remove(table_filepath)

    def _normalize_inputs(inp):
        if type(inp) == str:
            return [inp] * len(data_map_filepaths)
        if (type(inp) == list or type(inp) == tuple) and len(inp) == 1:
            return inp * len(data_map_filepaths)
        if type(inp) == list or type(inp) == tuple:
            assert len(inp) == len(data_map_filepaths), 'Invalid input!'
            return inp
        else:
            raise RuntimeError('Invalid input!')

    types = _normalize_inputs(types)
    views = _normalize_inputs(views)
    groups = _normalize_inputs(groups)

    if use_abs_path:
        uri = data_map_filepaths
    else:
        uri = [os.path.relpath(p, os.path.split(table_filepath)[0]) for p in data_map_filepaths]

    table_dict = dict(
            uri=uri,
            type=types,
            view=views,
            group=groups
        )

    if affine is not None:
        table_dict['affine'] = [','.join([str(x) for x in a]) for a in affine]
    if name is not None:
        table_dict['name'] = _normalize_inputs(name)
    if blend is not None:
        table_dict['blend'] = _normalize_inputs(blend)
    if exclusive is not None:
        table_dict['exclusive'] = _normalize_inputs(exclusive)

    append_mobie_table(
        table_filepath,
        table_dict
    )


class MobieTableProject:

    def __init__(self, project_dirpath, verbose=False):

        self._project_dirpath = project_dirpath
        self._verbose = verbose
        self._mobie_table_path = get_mobie_table_path(self._project_dirpath)

        if not os.path.exists(project_dirpath):
            os.mkdir(project_dirpath)
            return

    def init_with_data_maps(
            self,
            data_map_filepaths,
            types='intensities',
            views='raw',
            groups='group0'
    ):
        assert len(os.listdir(self._project_dirpath)) == 0, 'Can only initialize a project in an empty directory'
        init_mobie_table(
            self._mobie_table_path,
            data_map_filepaths,
            types=types,
            views=views,
            groups=groups,
            verbose=self._verbose
        )

    def get_project_dirpath(self):
        return self._project_dirpath



