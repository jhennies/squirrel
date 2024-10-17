
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

