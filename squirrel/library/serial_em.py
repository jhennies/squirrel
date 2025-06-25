import numpy as np
import os


def navigator_file_to_dict(filepath):

    import re

    data = {}
    items = {}
    current_item = None

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):  # Ignore empty lines and comments
                continue

            # Match key-value pairs like "version = 2.00"
            match = re.match(r"(\w+)\s*=\s*(.+)", line)
            if match:
                key, value = match.groups()

                # Convert numerical values if possible
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass

                if current_item is None:
                    data[key] = value
                else:
                    items[current_item][key] = value
                continue

            # Match item sections like "[Item = 7-A]" or "[Item = empty]"
            # match = re.match(r"\[Item\s*=\s*([\w-]+)\]", line)
            match = re.match(r"\[Item\s*=\s*(.*?)\]", line)
            if match:
                current_item = match.group(1)
                items[current_item] = {}
                continue

    data['items'] = items
    return data


def navigator_dict_to_file(nav_dict, filepath):

    import json
    with open(filepath, mode='w') as f:
        json.dump(nav_dict, f, indent=2)


def get_map_items_by_map_file(nav_dict, endswith='_search.mrc'):

    out_items = {}

    for key, val in nav_dict['items'].items():
        if 'MapFile' in val:
            if val['MapFile'].endswith(endswith):
                out_items[key] = val

    return out_items


def get_map_items_by_map_file(nav_dict, endswith='_search.mrc'):

    out_items = {}

    for key, val in nav_dict['items'].items():
        if 'MapFile' in val:
            if val['MapFile'].endswith(endswith):
                out_items[key] = val

    return out_items


def get_stage_xyz_from_item(item_dict):

    if 'StageXYZ' in item_dict:
        return [float(x) for x in item_dict['StageXYZ'].split(' ')]


def get_raw_stage_xy_from_item(item_dict):

    if 'RawStageXY' in item_dict:
        return [float(x) for x in item_dict['RawStageXY'].split(' ')]  # [::-1]


def get_map_scale_matrix_from_item(item_dict):
    if 'MapScaleMat' in item_dict:
        mat = [float(x) for x in item_dict['MapScaleMat'].split(' ')]
        mat = np.array(mat).reshape([2, 2], order='C')
        return mat


def get_value_list_from_item(item_dict, name):
    if name in item_dict:
        return [float(x) for x in item_dict[name].split(' ')]


def get_value_from_item(item_dict, name):
    if name in item_dict:
        return float(item_dict[name])


def get_type_from_item(item):
    map_file = item['MapFile'].replace("\\", '/')
    if 'gridmap.st' in map_file:
        return 'grid'
    if '_search.mrc' in map_file:
        return 'search'
    if '_view.mrc' in map_file:
        return 'view'
    if '_record.mrc' in map_file:
        return 'record'
    raise RuntimeError(f'Map type could not be inferred from file name: {map_file}')


def get_map_scale_xy(map_item):
    return get_value_list_from_item(map_item, 'StageXYZ')[:2]


def stage_to_image_coords(stage_coords, scale_mat):
    """
    Convert stage coordinates (microns) to pixel coordinates on the map image.

    Parameters:
        stage_coords (list of tuples): Stage coordinates (microns).
        scale_mat (list): [a, b, c, d] from MapScaleMat.

    Returns:
        list of tuples: Image pixel coordinates.
    """
    a, b, c, d = scale_mat
    affine = np.array([[a, b],
                       [c, d]])

    stage_coords = np.asarray(stage_coords)
    # FIXME: I'm still not sure if I need to do affine or affine.T
    #  so far, it didn't do much of a difference, because the matrices were quite symmetric
    img_coords = stage_coords @ affine.T
    return [tuple(coord) for coord in img_coords]


def get_mdoc_from_map_filepath(map_type, filepath):
    if map_type == 'grid':
        return os.path.join(os.path.split(filepath)[0], "gridmap.st.mdoc")
    if map_type == 'search':
        return f"{'_'.join(filepath.split('_')[:-2])}.mrc.mdoc"
    if map_type == 'view':
        return f"{'_'.join(filepath.split('_')[:-1])}.mrc.mdoc"
    if map_type == 'record':
        return f"{'_'.join(filepath.split('_')[:-2])}.mrc.mdoc"
    raise ValueError(f'Invalid map_type: {map_type}')


def get_gridmap_filepath(nav_filepath):
    from glob import glob
    import os
    return glob(os.path.join(
        os.path.split(nav_filepath)[0],
        'gridmap_*.mrc'
    ))[0]


def get_searchmap_filepath(search_map_item, nav_filepath, binning=4, pad_zeros=0):
    from glob import glob
    import os
    map_filepath = search_map_item['MapFile'].replace("\\", '/')
    map_section = int(search_map_item['MapSection']) + 1
    map_section_str = ('{:0' + str(pad_zeros) + 'd}').format(map_section)
    return os.path.join(
        os.path.split(nav_filepath)[0],
        f'{os.path.splitext(os.path.split(map_filepath)[1])[0]}_{map_section_str}_bin{binning}.mrc'
    )


def get_view_map_filepath(view_map_item, nav_filepath, pad_zeros=0):
    from glob import glob
    import os
    map_filepath = view_map_item['MapFile'].replace("\\", '/')
    map_section = int(get_nav_item_sec_from_note(view_map_item)) + 1
    map_section_str = ('{:0' + str(pad_zeros) + 'd}').format(map_section)
    return os.path.join(
        os.path.split(nav_filepath)[0],
        f'{os.path.splitext(os.path.split(map_filepath)[1])[0]}_{map_section_str}.mrc'
    )


def get_record_map_filepath(record_map_item, nav_filepath, binning=4, pad_zeros=0):
    from glob import glob
    import os
    map_filepath = record_map_item['MapFile'].replace("\\", '/')
    map_section = int(get_nav_item_sec_from_note(record_map_item)) + 1
    map_section_str = ('{:0' + str(pad_zeros) + 'd}').format(map_section)
    return os.path.join(
        os.path.split(nav_filepath)[0],
        f'{os.path.splitext(os.path.split(map_filepath)[1])[0]}_{map_section_str}_bin{binning}.mrc'
    )


def get_map_filepath(nav_filepath, map_item=None, binning=1):
    map_type = get_type_from_item(map_item)
    if map_type == 'grid':
        fp = get_gridmap_filepath(nav_filepath)
    if map_type == 'search':
        for zero_padding in range(4):
            fp = get_searchmap_filepath(map_item, nav_filepath, binning=binning, pad_zeros=zero_padding)
            if os.path.exists(fp):
                return fp
    if map_type == 'view':
        for zero_padding in range(4):
            fp = get_view_map_filepath(map_item, nav_filepath, pad_zeros=zero_padding)
            if os.path.exists(fp):
                return fp
    if map_type == 'record':
        for zero_padding in range(4):
            fp = get_record_map_filepath(map_item, nav_filepath, binning=binning, pad_zeros=zero_padding)
            if os.path.exists(fp):
                return fp
    if os.path.exists(fp):
        return fp
    raise RuntimeError(f'No valid filepath found for {map_filepath}')



def get_nav_item_id_from_note(in_item):
    return in_item['Note'].split(' ')[0]


def get_nav_item_sec_from_note(in_item):
    return in_item['Note'].split(' - ')[0].split(' ')[-1]


def get_view_map_items_by_drawn_id(view_map_items, drawn_id, nav_dict_items, return_dict=False):
    if return_dict:
        return {k: v for k, v in view_map_items.items() if
                nav_dict_items[get_nav_item_id_from_note(v)]['DrawnID'] == drawn_id}
    return [k for k, v in view_map_items.items() if nav_dict_items[get_nav_item_id_from_note(v)]['DrawnID'] == drawn_id]


def assign_view_maps_to_search_map(view_map_items, search_map_items, nav_dict_items, return_items=False):

    map_ids = {k: v['MapID'] for k, v in search_map_items.items()}

    if return_items:

        return {
            k: get_view_map_items_by_drawn_id(view_map_items, v, nav_dict_items, return_dict=True)
            for k, v in map_ids.items()
        }

    return {k: get_view_map_items_by_drawn_id(view_map_items, v, nav_dict_items) for k, v in map_ids.items()}


def get_resolution_from_mdoc(filepath, unit='micrometer'):

    import re

    with open(filepath, 'r') as file:
        mdoc_content = file.read()
    pixel_spacing_match = re.search(r'PixelSpacing\s*=\s*([\d.]+)', mdoc_content)
    pixel_spacing = float(pixel_spacing_match.group(1))

    if unit == 'micrometer':
        return pixel_spacing / 10000
    if unit == 'nanometer':
        return pixel_spacing / 10
    if unit == 'angstrom':
        return pixel_spacing
    raise ValueError(f'Invalid unit: {unit}')


def get_resolution_of_nav_item(nav_item, mdoc_dirpath=None, unit='micrometer', map_type=None):

    import os
    map_filepath = nav_item['MapFile'].replace("\\", '/')
    if mdoc_dirpath is None:
        mdoc_filepath = f'{map_filepath}.mdoc'
    else:
        mdoc_filepath = os.path.join(
            mdoc_dirpath,
            f'{os.path.split(map_filepath)[1]}.mdoc'
        )

    return get_resolution_from_mdoc(mdoc_filepath, unit=unit)


def get_search_map_on_grid_info(nav_filepath, grid_map_img_bin=8, search_map_img_bin=4, verbose=False):

    from squirrel.library.serial_em import navigator_file_to_dict
    nav_dict = navigator_file_to_dict(nav_filepath)

    # Get relevant navigator items
    from squirrel.library.serial_em import get_map_items_by_map_file
    search_map_items = get_map_items_by_map_file(nav_dict, '_search.mrc')
    grid_map_items = get_map_items_by_map_file(nav_dict, 'gridmap.st')

    # Get resolution of grid map
    from squirrel.library.serial_em import get_resolution_of_nav_item
    grid_map_resolution = get_resolution_of_nav_item(
        grid_map_items[next(iter(grid_map_items))],
        os.path.split(nav_filepath)[0],
        unit='micrometer'
    )
    # Get the resolutions of the search maps (only to return it)
    search_map_resolutions = [
        get_resolution_of_nav_item(
            search_map_item,
            os.path.split(nav_filepath)[0],
            unit='micrometer'
        )
        for _, search_map_item in search_map_items.items()
    ]

    # Get the search map information
    from squirrel.library.serial_em import get_raw_stage_xy_from_item
    search_map_names = []
    search_map_stage_positions = []
    search_map_item_list = []
    search_map_scale_mats = []
    for k, v in search_map_items.items():
        search_map_names.append(str(v['MapSection'] + 1))
        search_map_stage_positions.append(get_raw_stage_xy_from_item(v))
        search_map_item_list.append(v)
        search_map_scale_mats.append(np.array(get_value_list_from_item(v, 'MapScaleMat')) / search_map_img_bin)

    search_map_stage_positions = np.array(search_map_stage_positions)
    stage_xy = np.array(get_value_list_from_item(grid_map_items[next(iter(grid_map_items))], 'StageXYZ'))[:2]

    # # Use the affine transform to get the transformed stage positions
    grid_map_binning = get_value_from_item(grid_map_items[next(iter(grid_map_items))], 'MapBinning')
    map_scale_mat = np.array(
        get_value_list_from_item(grid_map_items[next(iter(grid_map_items))], 'MapScaleMat')
    ) / grid_map_img_bin

    transformed_stage_positions = np.array(stage_to_image_coords(
        (np.array(search_map_stage_positions) - np.array(stage_xy)),
        map_scale_mat
    ))

    return_dict = dict(
        search_map_names=search_map_names,
        search_map_item_list=search_map_item_list,
        search_map_positions=transformed_stage_positions,
        search_map_resolutions=search_map_resolutions,
        search_map_scale_mats=search_map_scale_mats,
        search_map_stage_xys=search_map_stage_positions,
        grid_map_resolution=grid_map_resolution,
        grid_map_scale_mat=map_scale_mat,
        grid_map_stage_xy=stage_xy,
        grid_map_binning=grid_map_binning
    )

    if verbose:
        print(f'return_dict = {return_dict}')

    return return_dict


def get_view_on_search_map_info(
        nav_filepath,
        search_map_item,
        view_map_items,
        search_map_img_bin=4,
        view_map_img_bin=4,
        verbose=False
):

    # Get resolution of search maps
    from squirrel.library.serial_em import get_resolution_of_nav_item
    search_map_resolution = get_resolution_of_nav_item(
        search_map_item,
        os.path.split(nav_filepath)[0],
        unit='micrometer'
    )
    # Get the resolutions of the view maps (only to return it)
    view_map_resolutions = [
        get_resolution_of_nav_item(
            view_map_item,
            os.path.split(nav_filepath)[0],
            unit='micrometer'
        )
        for _, view_map_item in view_map_items.items()
    ]

    # Get the view map information
    from squirrel.library.serial_em import get_raw_stage_xy_from_item
    view_map_names = []
    view_map_stage_positions = []
    view_map_item_list = []
    view_map_scale_mats = []
    view_map_binnings = []
    for k, v in view_map_items.items():
        view_map_names.append(str(v['MapSection'] + 1))
        view_map_stage_positions.append(get_raw_stage_xy_from_item(v))
        view_map_item_list.append(v)
        view_map_scale_mats.append(np.array(get_value_list_from_item(v, 'MapScaleMat')) / view_map_img_bin)
        view_map_binnings.append(get_value_from_item(v, 'MapBinning'))

    search_map_binning = get_value_from_item(search_map_item, 'MapBinning')
    stage_xy = np.array(get_value_list_from_item(search_map_item, 'StageXYZ'))[:2]
    map_scale_mat = np.array(get_value_list_from_item(search_map_item, 'MapScaleMat')) / search_map_img_bin

    transformed_stage_positions = np.array(stage_to_image_coords(
        (np.array(view_map_stage_positions) - np.array(stage_xy)),
        map_scale_mat
    ))

    return_dict = dict(
        view_map_names=view_map_names,
        view_map_item_list=view_map_item_list,
        view_map_positions=transformed_stage_positions,
        view_map_resolutions=view_map_resolutions,
        view_map_scale_mats=view_map_scale_mats,
        view_map_stage_xys=view_map_stage_positions,
        view_map_binnings=view_map_binnings,
        search_map_resolution=search_map_resolution,
        search_map_scale_mat=map_scale_mat,
        search_map_stage_xy=stage_xy,
        search_map_binning=search_map_binning
    )

    if verbose:
        print(f'return_dict = {return_dict}')

    return return_dict


def get_all_view_on_search_map_infos(
        nav_filepath,
        search_map_img_bin=4,
        verbose=False
):

    from squirrel.library.serial_em import navigator_file_to_dict
    nav_dict = navigator_file_to_dict(nav_filepath)

    # Get relevant navigator items
    from squirrel.library.serial_em import get_map_items_by_map_file
    search_map_items = get_map_items_by_map_file(nav_dict, '_search.mrc')
    view_map_items = get_map_items_by_map_file(nav_dict, '_view.mrc')

    # Figure out which view maps belong to which search map
    from squirrel.library.serial_em import assign_view_maps_to_search_map
    view_to_search_map_items = assign_view_maps_to_search_map(view_map_items, search_map_items, nav_dict['items'])

    search_map_infos = []

    for search_map_id, view_map_ids in view_to_search_map_items.items():
        if view_map_ids:
            search_map_infos.append(get_view_on_search_map_info(
                nav_filepath,
                search_map_items[search_map_id],
                {k: v for k, v in view_map_items.items() if k in view_map_ids},
                search_map_img_bin=search_map_img_bin,
                verbose=verbose
            ))
        else:
            print(f'Search map ID: {search_map_id} does not have corresponding views.')

    return search_map_infos


class Navigator:

    SEARCH_STRINGS = dict(
        record='_record.mrc',
        view='_view.mrc',
        search='_search.mrc',
        grid='gridmap.st'
    )

    def __init__(self, filepath, record_bin=1, view_bin=1, search_bin=4, grid_bin=8):

        self.record_bin = record_bin
        self.view_bin = view_bin
        self.search_bin = search_bin
        self.grid_bin = grid_bin

        self.binnings = dict(
            grid=grid_bin,
            search=search_bin,
            view=view_bin,
            record=record_bin
        )

        self.filepath = filepath
        self.nav_dict = navigator_file_to_dict(filepath)

        self.grid_map_items_dict = self.get_map_items_dict('grid')
        self.search_map_items_dict = self.get_map_items_dict('search')
        self.view_map_items_dict = self.get_map_items_dict('view')
        self.record_map_items_dict = self.get_map_items_dict('record')

        self.grid_map_item, self.grid_map_item_key = self.get_grid_map_item()
        (
            self.search_map_items,
            self.view_map_items,
            self.search_map_item_keys,
            self.view_map_item_keys,
            self.record_map_items,
            self.record_map_item_keys
        ) = self.get_search_and_view_map_items()

        self.grid_map_filepath, self.grid_mdoc_filepath = self.get_map_filepaths('grid')
        self.grid_map_filepath = self.grid_map_filepath[0]
        self.grid_mdoc_filepath = self.grid_mdoc_filepath[0]
        self.search_map_filepaths, self.search_mdoc_filepaths = self.get_map_filepaths('search')
        self.view_map_filepaths, self.view_mdoc_filepaths = self.get_map_filepaths('view')
        self.record_map_filepaths, self.record_mdoc_filepaths = self.get_map_filepaths('record')

    def get_map_names(self, map_type):
        pass

    def get_map_shapes(self, map_type):
        binning = dict(
            grid=self.grid_bin,
            search=self.search_bin,
            view=self.view_bin,
            record=self.record_bin
        )
        def _this(item):
            bin_factor = get_value_from_item(item, 'MapBinning') / (get_value_from_item(item, 'MontBinning') * binning[map_type])
            return (np.array(get_value_list_from_item(item, 'MapWidthHeight')) * bin_factor).astype(int)
        return self._get_values(map_type, _this)

    def get_map_items_dict(self, map_type):

        map_items = get_map_items_by_map_file(self.nav_dict, self.SEARCH_STRINGS[map_type])

        if map_type != 'view' and map_type != 'record':
            return map_items

        return assign_view_maps_to_search_map(
            map_items, self.search_map_items_dict, self.nav_dict['items'], return_items=True
        )

    def get_grid_map_item(self):
        key = next(iter(self.grid_map_items_dict))
        return self.grid_map_items_dict[key], key

    def get_search_and_view_map_items(self):
        def sort_key(item):
            import re
            match = re.match(r"(\d+)(.*)", item)
            if match:
                number = int(match.group(1))
                suffix = match.group(2)
                return (number, suffix)
            return (float('inf'), item)

        sorted_search_map_keys = sorted(list(self.search_map_items_dict.keys()), key=sort_key)

        search_map_items = []
        view_map_items = []
        record_map_items = []
        search_map_item_keys = []
        view_map_item_keys = []
        record_map_item_keys = []
        for key in sorted_search_map_keys:
            if len(self.view_map_items_dict[key]) > 0:
                search_map_items.append(self.search_map_items_dict[key])
                search_map_item_keys.append(key)
                sorted_view_map_keys = sorted(list(self.view_map_items_dict[key].keys()), key=sort_key)
                sorted_record_map_keys = sorted(list(self.record_map_items_dict[key].keys()), key=sort_key)
                view_map_items.append([self.view_map_items_dict[key][vm_key] for vm_key in sorted_view_map_keys])
                view_map_item_keys.append(sorted_view_map_keys)
                record_map_items.append([self.record_map_items_dict[key][rm_key] for rm_key in sorted_record_map_keys])
                record_map_item_keys.append(sorted_record_map_keys)
            else:
                print(f'Search map ID: {key} does not have corresponding views.')

        return search_map_items, view_map_items, search_map_item_keys, view_map_item_keys, record_map_items, record_map_item_keys

    def get_map_positions(self, map_type):
        pass

    def get_map_resolutions(self, map_type):
        def _this(item):
            map_type = get_type_from_item(item)
            binning = self.binnings[map_type]
            fp = get_map_filepath(self.filepath, map_item=item, binning=binning)
            mdoc_fp = get_mdoc_from_map_filepath(map_type, fp)
            return get_resolution_from_mdoc(mdoc_fp, unit='micrometer')
        return self._get_values(map_type, _this)

    def get_map_scale_matrices(self, map_type):
        return self._get_values(map_type, get_map_scale_matrix_from_item)

    def get_map_scale_xys(self, map_type):
        return self._get_values(map_type, get_map_scale_xy)

    def get_map_filepaths(self, map_type):
        # FIXME Use self._get_values
        if map_type == 'grid':
            fp = get_map_filepath(self.filepath, self.grid_map_item)
            mdoc_fp = get_mdoc_from_map_filepath(map_type, fp)
            return [fp], [mdoc_fp]
        if map_type == 'search':
            fps = []
            mdoc_fps = []
            for x in self.search_map_items:
                fps.append(get_map_filepath(self.filepath, x, self.search_bin))
                mdoc_fps.append(get_mdoc_from_map_filepath(map_type, fps[-1]))
            return fps, mdoc_fps
        if map_type == 'view':
            fps = []
            mdoc_fps = []
            for x in self.view_map_items:
                this_fps = []
                this_mdoc_fps = []
                for y in x:
                    this_fps.append(get_map_filepath(self.filepath, y))
                    this_mdoc_fps.append(get_mdoc_from_map_filepath(map_type, this_fps[-1]))
                fps.append(this_fps)
                mdoc_fps.append(this_mdoc_fps)
            return fps, mdoc_fps
        if map_type == 'record':
            fps = []
            mdoc_fps = []
            for x in self.record_map_items:
                this_fps = []
                this_mdoc_fps = []
                for y in x:
                    this_fps.append(get_map_filepath(self.filepath, y, binning=self.record_bin))
                    this_mdoc_fps.append(get_mdoc_from_map_filepath(map_type, this_fps[-1]))
                fps.append(this_fps)
                mdoc_fps.append(this_mdoc_fps)
            return fps, mdoc_fps
        raise ValueError(f'Invalid map_type: {map_type}')

    def get_map_full_affines(self, map_type, flatten=False, invert=False, full_square=False):
        """
        This function returns the full affine transformation that maps stage coordinates to the image pixels.
        Note that this is not necessarily identical with the MapAffine entry in the navigator as (a) the image binning
        (parameter when instanciating this class) and (b) the difference of MapBinning and MontBinning is applied here
        to the MapAffine.

        :param map_type:
        :param flatten:
        :param invert:
        :param full_square:
        :return:
        """

        def _get_affine():

            bin_factor = map_binning / mont_binning * map_resolution
            affine = np.array([
                [mat[0, 0] * bin_factor, mat[0, 1] * bin_factor, 0, img_shp[0] / 2 * (map_resolution * bin)],
                [mat[1, 0] * bin_factor, mat[1, 1] * bin_factor, 0, img_shp[1] / 2 * (map_resolution * bin)],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            if invert:
                affine = np.linalg.inv(affine)
            if not full_square:
                affine = affine[:3]
            affine[0, 3] += xy[0]
            affine[1, 3] += xy[1]
            if flatten:
                affine = affine.flatten()

            return affine

        bin = 1
        if map_type == 'grid':
            bin = self.grid_bin
        if map_type == 'search':
            bin = self.search_bin
        if map_type == 'view':
            bin = self.view_bin
        if map_type == 'record':
            bin = self.record_bin

        scale_mats = self.get_map_scale_matrices(map_type)
        xys = self.get_map_scale_xys(map_type)

        img_shapes = self.get_map_shapes(map_type)

        map_binnings = self.get_map_binnings(map_type)
        mont_binnings = self.get_mont_binnings(map_type)
        map_resolutions = self.get_map_resolutions(map_type)

        affines = []

        for idx, scale_mat in enumerate(scale_mats):

            if map_type != 'view' and map_type != 'record':

                xy = xys[idx]
                img_shp = img_shapes[idx]
                map_binning = map_binnings[idx]
                mont_binning = mont_binnings[idx]
                map_resolution = map_resolutions[idx]

                mat = scale_mat

                affine = _get_affine()

            else:

                this_xys = xys[idx]
                this_img_shapes = img_shapes[idx]
                this_map_binnings = map_binnings[idx]
                this_mont_binnings = mont_binnings[idx]
                this_map_resolutions = map_resolutions[idx]

                affine = []
                for jdx, mat in enumerate(scale_mat):
                    xy = this_xys[jdx]
                    img_shp = this_img_shapes[jdx]
                    map_binning = this_map_binnings[jdx]
                    mont_binning = this_mont_binnings[jdx]
                    map_resolution = this_map_resolutions[jdx]

                    affine.append(_get_affine())

            affines.append(affine)

        return affines

    def get_map_full_affines_to_grid_map(self, map_type, stage_coordinate_system=False):

        affine = self.get_map_full_affines(map_type, flatten=False, invert=True, full_square=True)
        grid_affine = self.get_map_full_affines('grid', flatten=False, invert=False, full_square=True)

        if stage_coordinate_system:
            # Aligned to stage coordinate system
            if map_type == 'grid':
                return [(affine[0])[:3].flatten()]
            if map_type == 'search':
                return [x[:3].flatten() for x in affine]
            if map_type in ['view', 'record']:
                return [[y[:3].flatten() for y in x] for x in affine]

            raise ValueError(f'Invalid map_type! {map_type}')

        # Aligned to grid map
        if map_type == 'grid':
            return [(grid_affine[0] @ affine[0])[:3].flatten()]
        if map_type == 'search':
            return [(grid_affine[0] @ x)[:3].flatten() for x in affine]
        if map_type in ['view', 'record']:
            return [[(grid_affine[0] @ y)[:3].flatten() for y in x] for x in affine]

        raise ValueError(f'Invalid map_type! {map_type}')

    def get_map_binnings(self, map_type):
        def _this(item):
            return get_value_from_item(item, 'MapBinning')
        return self._get_values(map_type, _this)

    def get_mont_binnings(self, map_type):
        def _this(item):
            return get_value_from_item(item, 'MontBinning')
        return self._get_values(map_type, _this)

    def _get_values(self, map_type, func):
        if map_type == 'grid':
            return [func(self.grid_map_item)]
        if map_type == 'search':
            return [func(x) for x in self.search_map_items]
        if map_type == 'view':
            return [[func(y) for y in x] for x in self.view_map_items]
        if map_type == 'record':
            return [[func(y) for y in x] for x in self.record_map_items]

        raise ValueError(f'Invalid map_type! {map_type}')
