
import os
import numpy as np


def export_rois_with_mobie_table_workflow(
        table_filepath,
        map_dirpath,
        target_dirpath,
        map_resolution=None,
        output_filetype='tif',
        label_ids=None,
        verbose=False,
):
    if verbose:
        print(f'table_filepath = {table_filepath}')
        print(f'map_dirpath = {map_dirpath}')
        print(f'target_dirpath = {target_dirpath}')
        print(f'map_resolution = {map_resolution}')
        print(f'output_filetype = {output_filetype}')
        print(f'label_ids = {label_ids}')

    assert map_resolution is not None, 'Resolution of the map must be specified!'

    def _load_table(table_fp):
        import pandas as pd
        table = pd.read_csv(table_fp, delimiter='\t')
        table.set_index('label_id', inplace=True)
        return table

    def _get_position_px(table, idx, res):
        row = table.loc[idx]
        x = row['bb_min_x']
        y = row['bb_min_y']
        z = row['bb_min_z']
        w = row['bb_max_x'] - x
        h = row['bb_max_y'] - y
        d = row['bb_max_z'] - z
        return np.array((z, y, x)) / np.array(res), np.array((d, h, w)) / np.array(res)

    write_func = None
    if output_filetype == 'tif':
        from squirrel.library.io import write_tif_stack

        def _write_tif_stack(data, idx, x, y, z):
            write_tif_stack(data, os.path.join(target_dirpath, 'label_{:05d}_x{}_y{}_z{}'.format(idx, x, y, z)),
                            'slice_{:04d}.tif')

        write_func = _write_tif_stack

    if write_func is None:
        raise ValueError(f'Invalid output filetype: {output_filetype}')

    table = _load_table(table_filepath)

    # from squirrel.library.io import load_data_handle
    # h, shape = load_data_handle(map_dirpath, key=None, pattern=None)

    for idx in label_ids:

        zyx, whd = _get_position_px(table, idx, map_resolution)
        z, y, x = np.array(zyx).astype(int)
        w, h, d = np.array(whd).astype(int)

        data = h[z:z+d, y:y+h, x:x+w]

        write_func(data, idx, x, y, z)

