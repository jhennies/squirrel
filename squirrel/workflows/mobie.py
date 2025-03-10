
import os
import numpy as np


def _get_position_px(table, idx, res):
    row = table.loc[idx]
    x = row['bb_min_x']
    y = row['bb_min_y']
    z = row['bb_min_z']
    w = row['bb_max_x'] - x
    h = row['bb_max_y'] - y
    d = row['bb_max_z'] - z
    return np.array((z, y, x)) / np.array(res), np.array((d, h, w)) / np.array(res), [z, y, x]


def _get_data(data_h, idx, resolution, table, verbose=False):
    zyx, dhw, zyx_res = _get_position_px(table, idx, resolution)
    z, y, x = np.array(zyx).astype(int)
    d, h, w = np.array(dhw).astype(int)

    if verbose:
        print(f'zyx = {zyx}')
        print(f'dhw = {dhw}')

    return data_h[z:z + d, y:y + h, x:x + w], zyx_res


def _apply_mask(map_data, mask_h, idx, map_resolution, mask_resolution, table, verbose=False):
    mask_data, _ = _get_data(mask_h, idx, mask_resolution, table, verbose=verbose)

    if mask_resolution != map_resolution:
        from squirrel.library.scaling import scale_image_nearest
        mask_data = scale_image_nearest(mask_data, np.array(mask_resolution) / np.array(map_resolution))

    print(f'map_data.shape = {map_data.shape}')
    print(f'mask_data.shape = {mask_data.shape}')
    # mask_data_pad = np.zeros(map_data.shape)
    # mask_data_pad[:mask_data.shape[0], :mask_data.shape[1], :mask_data.shape[2]] = mask_data
    # map_data[mask_data_pad != idx] = 0

    map_data = map_data[:mask_data.shape[0], :mask_data.shape[1], :mask_data.shape[2]]
    print(f'map_data.shape = {map_data.shape}')
    map_data[mask_data != idx] = 0
    return map_data


def _cast_func(sl, sl_idx, label_mapping):
    print(f'sl_idx = {sl_idx}')
    u = np.unique(sl)
    if len(u) > 1 or u[0] != 0:
        lm = {key: label_mapping[key] for key in list(u) if key in label_mapping}
        map_func = np.vectorize(lm.get)
        return map_func(sl)
    return sl


def _cast_dtype(map_data, n_workers=1):
    from squirrel.library.data import get_optimal_dtype
    print(f'Casting dtype:')
    print(f'Getting label mapping ...')
    label_list = np.unique(map_data)
    label_mapping = dict(zip(label_list, range(len(label_list))))
    print(f'Finding optimal dtype ...')
    dtype = get_optimal_dtype(len(label_list))
    print(f'Mapping the data ...')
    if n_workers == 1:
        for sl_idx, sl in enumerate(map_data):
            print(f'sl_idx = {sl_idx}')
            u = np.unique(sl)
            if len(u) > 1 or u[0] != 0:
                lm = {key: label_mapping[key] for key in list(u) if key in label_mapping}
                map_func = np.vectorize(lm.get)
                map_data[sl_idx, :] = map_func(sl)
    else:
        from multiprocessing import Pool
        with Pool(processes=n_workers) as p:
            tasks = [
                p.apply_async(
                    _cast_func,
                    (sl, sl_idx, label_mapping)
                )
                for sl_idx, sl in enumerate(map_data)
            ]
            map_data = np.array([task.get() for task in tasks])

    return map_data.astype(dtype)


def _run_for_label_id(
        idx, map_h, mask_h, map_resolution, mask_resolution, write_func, table,
        target_dirpath,
        n_workers=1,
        verbose=False
):
    if verbose:
        print(f'------------- Executing LabelID: {idx} --------------')
        print(f'Loading data ...')
    map_data, zyx = _get_data(map_h, idx, map_resolution, table, verbose=verbose)

    if verbose:
        print(f'data.shape = {map_data.shape}')

    if mask_h is not None:
        if verbose:
            print(f'Applying mask ...')
        map_data = _apply_mask(map_data, mask_h, idx, map_resolution, mask_resolution, table, verbose=verbose)

    if verbose:
        print(f'np.unique(map_data[s/2, :]) = {np.unique(map_data[int(map_data.shape[0] / 2), :])}')
        print(f'Casting dtype ...')
    map_data = _cast_dtype(map_data, n_workers=n_workers)

    if verbose:
        print(f'Writing result for label {idx} to {target_dirpath}...')
    write_func(map_data, idx, zyx[2], zyx[1], zyx[0], target_dirpath)
    if verbose:
        print(f'Done!')


def _write_tif_stack(data, idx, x, y, z, target_dirpath):
    from squirrel.library.io import write_tif_stack
    write_tif_stack(data, os.path.join(target_dirpath, 'label_{:05d}_x{}_y{}_z{}'.format(idx, x, y, z)),
                    slice_name='slice_{:04d}.tif')


def export_rois_with_mobie_table_workflow(
        table_filepath,
        map_dirpath,
        target_dirpath,
        map_key=None,
        map_resolution=None,
        mask_dirpath=None,
        mask_key=None,
        mask_resolution=None,
        output_filetype='tif',
        label_ids=None,
        exclude_ids=None,
        n_workers=1,
        verbose=False,
):
    if verbose:
        print(f'table_filepath = {table_filepath}')
        print(f'map_dirpath = {map_dirpath}')
        print(f'target_dirpath = {target_dirpath}')
        print(f'map_resolution = {map_resolution}')
        print(f'mask_dirpath = {mask_dirpath}')
        print(f'mask_key = {mask_key}')
        print(f'mask_resolution = {mask_resolution}')
        print(f'output_filetype = {output_filetype}')
        print(f'n_workers = {n_workers}')
        print(f'label_ids = {label_ids}')
        print(f'exclude_ids = {exclude_ids}')
        print(f'n_workers = {n_workers}')

    assert map_resolution is not None, 'Resolution of the map must be specified!'
    if mask_dirpath is not None:
        assert mask_resolution is not None, 'Resolution of the mask must be specified!'

    if not os.path.exists(target_dirpath):
        os.mkdir(target_dirpath)

    def _load_table(table_fp):
        import pandas as pd
        table = pd.read_csv(table_fp, delimiter='\t')
        table.set_index('label_id', inplace=True)
        return table

    write_func = None
    if output_filetype == 'tif':

        write_func = _write_tif_stack

    if write_func is None:
        raise ValueError(f'Invalid output filetype: {output_filetype}')

    table = _load_table(table_filepath)

    if label_ids is None:
        label_ids = np.array(table.loc[:].index.tolist()).astype(int)
    if exclude_ids is not None:
        label_ids = np.array([x for x in label_ids if x not in exclude_ids])

    if verbose:
        print(f'label_ids = {label_ids}')

    mask_h = None
    mask_shape = None
    from squirrel.library.io import load_data_handle
    map_h, shape = load_data_handle(map_dirpath, key=map_key, pattern=None)
    if mask_dirpath is not None:
        mask_h, mask_shape = load_data_handle(mask_dirpath, key=mask_key, pattern=None)

    if True:  # n_workers == 1:
        for idx in label_ids:
            _run_for_label_id(
                idx, map_h, mask_h, map_resolution, mask_resolution, write_func, table,
                target_dirpath,
                n_workers=n_workers,
                verbose=verbose
            )

    else:

        from multiprocessing import Pool
        with Pool(processes=n_workers) as p:
            tasks = [
                p.apply_async(
                    _run_for_label_id,
                    (idx, map_h, mask_h, map_resolution, mask_resolution, write_func, table, target_dirpath, verbose)
                )
                for idx in label_ids
            ]
            [task.get() for task in tasks]


        # map_data, x, y, z = _get_data(map_h, idx, map_resolution)
        #
        # if verbose:
        #     print(f'data.shape = {map_data.shape}')
        #
        # if mask_h is not None:
        #     map_data = _apply_mask(map_data, mask_h, idx, mask_resolution)
        #
        # map_data = _cast_dtype(map_data)
        #
        # write_func(map_data, idx, x, y, z)

