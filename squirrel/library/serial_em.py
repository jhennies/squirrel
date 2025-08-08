import numpy as np
from pathlib import Path

# ______________________________________________________________________________________________________________________
# Functions of the original linkmaps workflow

def navigator_file_to_dict(filepath):

    def get_unique_key(base_key, existing_keys):
        """Generate a unique key by appending -1, -2, etc. if necessary."""
        if base_key not in existing_keys:
            return base_key
        i = 1
        while f"{base_key}-{i}" in existing_keys:
            i += 1
        return f"{base_key}-{i}"

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
                base_item = match.group(1)
                current_item = get_unique_key(base_item, items)
                items[current_item] = {}
                continue

    data['items'] = items
    return data


def get_map_items_by_map_file(nav_dict, endswith='_search.mrc'):

    out_items = {}

    for key, val in nav_dict['items'].items():
        if 'MapFile' in val:
            if val['MapFile'].endswith(endswith):
                out_items[key] = val

    return out_items


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

    map_filepath = Path(nav_item['MapFile'].replace("\\", '/'))
    if mdoc_dirpath is None:
        mdoc_filepath = map_filepath.parent / f'{map_filepath.name}.mdoc'
    else:
        mdoc_filepath = Path(mdoc_dirpath) / f'{map_filepath.name}.mdoc'

    return get_resolution_from_mdoc(mdoc_filepath, unit=unit)


def get_raw_stage_xy_from_item(item_dict):

    if 'RawStageXY' in item_dict:
        return [float(x) for x in item_dict['RawStageXY'].split(' ')]  # [::-1]


def get_value_list_from_item(item_dict, name):
    if name in item_dict:
        return [float(x) for x in item_dict[name].split(' ')]


def get_value_from_item(item_dict, name):
    if name in item_dict:
        return float(item_dict[name])


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


def get_search_map_on_grid_info(nav_filepath, grid_map_img_bin=8, search_map_img_bin=4, verbose=False):

    nav_filepath = Path(nav_filepath)

    nav_dict = navigator_file_to_dict(nav_filepath)

    # Get relevant navigator items
    search_map_items = get_map_items_by_map_file(nav_dict, '_search.mrc')
    grid_map_items = get_map_items_by_map_file(nav_dict, 'gridmap.st')

    # Get resolution of grid map
    grid_map_resolution = get_resolution_of_nav_item(
        grid_map_items[next(iter(grid_map_items))],
        nav_filepath.parent,
        unit='micrometer'
    )
    # Get the resolutions of the search maps (only to return it)
    search_map_resolutions = [
        get_resolution_of_nav_item(
            search_map_item,
            nav_filepath.parent,
            unit='micrometer'
        )
        for _, search_map_item in search_map_items.items()
    ]

    # Get the search map information
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


def get_gridmap_filepath(nav_filepath, extension='mrc'):
    return list(Path(nav_filepath).parent.glob(f'gridmap_*.{extension}'))[0]


def get_nav_item_id_from_note(in_item):
    return in_item['Note'].split(' ')[0]


def get_view_map_items_by_drawn_id(view_map_items, drawn_id, nav_dict_items, return_dict=False):
    if return_dict:
        return {k: v for k, v in view_map_items.items() if
                nav_dict_items[get_nav_item_id_from_note(v)]['DrawnID'] == drawn_id}

    return_items = []
    for k, v in view_map_items.items():
        nav_item = nav_dict_items[get_nav_item_id_from_note(v)]
        if 'DrawnID' not in nav_item:
            print(f'Warning: "DrawnID" entry not in navigator item {get_nav_item_id_from_note(v)}')
            continue
        if nav_dict_items[get_nav_item_id_from_note(v)]['DrawnID'] == drawn_id:
            return_items.append(k)
    return return_items


def assign_view_maps_to_search_map(view_map_items, search_map_items, nav_dict_items, return_items=False):

    map_ids = {k: v['MapID'] for k, v in search_map_items.items()}

    if return_items:

        return {
            k: get_view_map_items_by_drawn_id(view_map_items, v, nav_dict_items, return_dict=True)
            for k, v in map_ids.items()
        }

    return {k: get_view_map_items_by_drawn_id(view_map_items, v, nav_dict_items) for k, v in map_ids.items()}


def get_view_on_search_map_info(
        nav_filepath,
        search_map_item,
        view_map_items,
        search_map_img_bin=4,
        view_map_img_bin=4,
        verbose=False
):

    nav_filepath = Path(nav_filepath)

    # Get resolution of search maps
    from squirrel.library.serial_em import get_resolution_of_nav_item
    search_map_resolution = get_resolution_of_nav_item(
        search_map_item,
        nav_filepath.parent,
        unit='micrometer'
    )
    # Get the resolutions of the view maps (only to return it)
    view_map_resolutions = [
        get_resolution_of_nav_item(
            view_map_item,
            nav_filepath.parent,
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


def get_searchmap_filepath(search_map_item, nav_filepath, binning=4, pad_zeros=0, extension='mrc'):
    map_filepath = Path(search_map_item['MapFile'].replace("\\", '/'))
    map_section = int(search_map_item['MapSection']) + 1
    map_section_str = ('{:0' + str(pad_zeros) + 'd}').format(map_section)
    return Path(nav_filepath).parent / f'{map_filepath.stem}_{map_section_str}_bin{binning}.{extension}'

# ______________________________________________________________________________________________________________________


def get_filepath_from_nav_item(nav_filepath, map_item, item_id):
    map_filepath = Path(map_item[item_id].replace("\\", '/'))
    return Path(nav_filepath).parent / map_filepath.name


def get_map_filepath_from_nav_item(nav_filepath, map_item):
    return get_filepath_from_nav_item(nav_filepath, map_item, 'MapFile')


def get_map_items_by_glob(nav_dict, nav_filepath, glob='*.mrc'):
    from fnmatch import fnmatch

    out_items = {}
    test_fp = Path(nav_filepath).parent / glob

    for key, val in nav_dict['items'].items():
        if 'MapFile' in val:
            item_fp = get_map_filepath_from_nav_item(nav_filepath, val)

            if fnmatch(item_fp, test_fp):
                out_items[key] = val

    return out_items


def get_mdoc_filepath(mrc_filepath):
    mrc_filepath = Path(mrc_filepath)
    return mrc_filepath.parent / f'{mrc_filepath.name}.mdoc'


def get_map_scale_xy(map_item):
    return get_value_list_from_item(map_item, 'StageXYZ')[:2]


def get_map_scale_matrix_from_item(item_dict):
    if 'MapScaleMat' in item_dict:
        mat = [float(x) for x in item_dict['MapScaleMat'].split(' ')]
        mat = np.array(mat).reshape([2, 2], order='C')
        return mat
    return None


def get_map_shape(map_item, binning):
    bin_factor = get_value_from_item(map_item, 'MapBinning') / (
                get_value_from_item(map_item, 'MontBinning') * binning)
    return (np.array(get_value_list_from_item(map_item, 'MapWidthHeight')) * bin_factor).astype(int)


def get_contrast_limits_from_map(
        map_fp
):
    from squirrel.library.data import get_contrast_limits
    import mrcfile
    with mrcfile.open(map_fp, permissive=True) as mrc:
        img = mrc.data
    return get_contrast_limits(img)


class Navigator:

    def __init__(
            self, filepath,
            map_types: list[str] = None,
            map_hierarchy: list[str] = None,
            search_strings: dict[str, str] = None,
            map_binnings: dict[str, int] = None,
            verbose: bool = False
    ):

        self.verbose = verbose
        self.map_types = map_types if map_types is not None else ['grid']
        self.search_strings = search_strings if search_strings is not None else dict(grid='gridmap.st')
        self.map_binnings = map_binnings if map_binnings is not None else dict(zip(self.map_types, [1] * len(self.map_types)))

        assert len(self.search_strings) == len(self.map_types), 'Number of search strings must match number of map types'
        assert len(self.map_binnings) == len(self.map_types), 'Number of map binnings must match number of map types'

        self.filepath = Path(filepath)
        self.nav_dict = navigator_file_to_dict(filepath)

        # It is important to add the items like this because some items might depend on the earlier added ones being
        #   present already
        self.map_items_dict = dict()
        for map_type in self.map_types:
            self.map_items_dict[map_type] = self._get_map_items_dict(map_type)

        self.map_filepaths = {map_type: self._get_map_filepaths(map_type) for map_type in self.map_types}

        self.mdoc_filepaths = {map_type: self._get_mdoc_filepaths(map_type) for map_type in self.map_types}

        self.map_contrast_limits = {map_type: None for map_type in self.map_types}

        self.map_sec_ids = {map_type: self._get_map_sec_ids(map_type) for map_type in self.map_types}

        self.map_hierarchy = map_hierarchy
        self._build_key_hierarchy()

        self._count = [0] * len(self.map_types)

    @staticmethod
    def _sort_key(item):
        import re
        match = re.match(r"(\d+)(.*)", item)
        if match:
            number = int(match.group(1))
            suffix = match.group(2)
            return number, suffix
        return float('inf'), item

    def _function_on_property(self, map_type, prop_name, func, **kwargs):
        try:
            return {k: func(map_type=map_type, key=k, item=v, **kwargs) for k, v in getattr(self, prop_name)[map_type].items()}
        except TypeError:
            return {k: func(v, **kwargs) for k, v in getattr(self, prop_name)[map_type].items()}

    def _get_map_items_dict(self, map_type):
        return get_map_items_by_glob(self.nav_dict, self.filepath, self.search_strings[map_type])

    def _get_map_filepath(self, item):
        return get_map_filepath_from_nav_item(self.filepath, item)

    def _get_map_filepaths(self, map_type):
        return self._function_on_property(map_type, 'map_items_dict', self._get_map_filepath)

    def _get_mdoc_filepaths(self, map_type):
        return self._function_on_property(map_type, 'map_filepaths', get_mdoc_filepath)

    def _get_map_resolution(self, map_type, key):
        mdoc_fp = self.mdoc_filepaths[map_type][key]
        return get_resolution_from_mdoc(mdoc_fp, unit='micrometer')

    def _get_affine(self, map_type, key, item, binning, invert=False, full_square=False, flatten=False):

        xy = get_map_scale_xy(item)
        mat = get_map_scale_matrix_from_item(item)
        img_shp = get_map_shape(item, binning)
        map_binning = get_value_from_item(item, 'MapBinning')
        mont_binning = get_value_from_item(item, 'MontBinning')
        map_resolution = self._get_map_resolution(map_type, key)

        bin_factor = map_binning / mont_binning * map_resolution
        affine = np.array([
            [mat[0, 0] * bin_factor, mat[0, 1] * bin_factor, 0, img_shp[0] / 2 * (map_resolution * binning)],
            [mat[1, 0] * bin_factor, mat[1, 1] * bin_factor, 0, img_shp[1] / 2 * (map_resolution * binning)],
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

    def _get_map_full_affine(self, map_type, key, item, binning=1, apply_affine=None, invert=False, full_square=False, flatten=False):

        if apply_affine is not None:
            affine = self._get_affine(map_type, key, item, binning, invert=invert, full_square=full_square, flatten=False)
            if flatten:
                return (apply_affine @ affine).flatten()
            return apply_affine @ affine

        return self._get_affine(map_type, key, item, binning, invert=invert, full_square=full_square, flatten=flatten)

    def get_grid_key(self):
        return next(iter(self.map_items_dict['grid']))

    def get_grid_item(self):
        key = self.get_grid_key()
        return self.map_items_dict['grid'][key]

    def get_map_full_affines(self, map_type, stage_coordinate_system=False):

        kwargs = dict(
            apply_affine=(
                None if stage_coordinate_system else
                self._get_map_full_affine('grid', self.get_grid_key(), self.get_grid_item(), self.map_binnings['grid'], invert=False, full_square=True, flatten=False)
            ),
            binning=self.map_binnings[map_type],
            invert=True, full_square=True, flatten=True
        )
        return self._function_on_property(map_type, 'map_items_dict', self._get_map_full_affine, **kwargs)

    @staticmethod
    def _get_map_sec_id(map_item):
        import re
        match = re.search(r'Sec (\d+)', map_item['Note'])
        return int(match.group(1)) if match else None

    def _get_map_sec_ids(self, map_type):
        return self._function_on_property(map_type, 'map_items_dict', self._get_map_sec_id)

    def get_contrast_limits(self, map_type):
        if self.map_contrast_limits[map_type] is None:
            print(f'Computing contrast limits for {map_type}')
            self.map_contrast_limits[map_type] = self._function_on_property(map_type, 'map_filepaths', get_contrast_limits_from_map)
        return self.map_contrast_limits[map_type]

    def _get_key_dependencies_for_map_type(self, map_type: str) -> list:
        if self.map_hierarchy is None:
            return list(self.map_items_dict[map_type].keys())
        raise NotImplementedError('map_hierarchy != None is not implemented for the base class! '
                                  'Classes that need this must implement the functionality '
                                  'and thus override this function.')

    def _build_key_hierarchy(self):
        self.key_hierarchy = dict()
        ordered_map_types = self.map_hierarchy if self.map_hierarchy is not None else self.map_types
        for map_type in ordered_map_types:
            self.key_hierarchy[map_type] = self._get_key_dependencies_for_map_type(map_type)

    def get_property_by_hierarchy(self, map_type, property, parent_key):

        def find_index_path(nested, target, path=None):
            if path is None:
                path = []

            if isinstance(nested, list):
                for i, item in enumerate(nested):
                    result = find_index_path(item, target, path + [i])
                    if result is not None:
                        return result
            else:
                if nested == target:
                    return path

            return None

        parent_map_type = self.map_hierarchy[self.map_hierarchy.index(map_type) - 1]
        index_path = find_index_path(self.key_hierarchy[parent_map_type], parent_key)

        this_keys = self.key_hierarchy[map_type]
        for idx in index_path:
            this_keys = this_keys[idx]

        return [property[x] for x in this_keys]

    def _key_generator(self):

        def loop_func(hierarchy, this_key_hierarchy, idx_path = None):
            this_hierarchy = hierarchy.pop(0)

            for idx, item in enumerate(this_key_hierarchy):
                this_idx_path = [idx] if idx_path is None else idx_path + [idx]
                yield this_hierarchy, item, this_idx_path

                if hierarchy:

                    this_child_key_hierarchy = self.key_hierarchy[hierarchy[0]]
                    try:
                        for jdx in this_idx_path:
                            this_child_key_hierarchy = this_child_key_hierarchy[jdx]
                    except IndexError:
                        continue
                    yield from loop_func(hierarchy.copy(), this_child_key_hierarchy, this_idx_path)

        yield from loop_func(self.map_hierarchy.copy(), self.key_hierarchy['grid'])

    def __iter__(self):
        return self._key_generator()


class SingleParticleNavigator(Navigator):

    MAP_TYPES = [
        'grid',
        'search',
        'view',
        'record'
    ]

    SEARCH_STRINGS = dict(
        record='*_record.mrc',
        view='*_view.mrc',
        search='*_search.mrc',
        grid='gridmap.st'
    )

    def __init__(self, filepath: str, record_bin: int = 1, view_bin: int = 1, search_bin: int = 4, grid_bin: int = 8):

        super().__init__(
            filepath,
            map_types=self.MAP_TYPES,
            map_hierarchy=self.MAP_TYPES,
            search_strings=self.SEARCH_STRINGS,
            map_binnings=dict(
                record=record_bin,
                view=view_bin,
                search=search_bin,
                grid=grid_bin
            )
        )

    @staticmethod
    def _assign_record_maps_to_view_maps(record_items, view_items):
        rec_maps_to_view_maps = dict()
        for view_k, view_v in view_items.items():
            vidx = view_v['Note'].split(' ')[0]
            for rec_k, rec_v in record_items.items():
                ridx = rec_v['Note'].split(' ')[0]
                if vidx == ridx:
                    rec_maps_to_view_maps[view_k] = rec_k
                    break
        return rec_maps_to_view_maps

    def _get_grid_map_filepath(self, item):
        fp = get_gridmap_filepath(self.filepath)
        return fp

    def _get_search_map_filepath(self, item):
        for zero_padding in range(4):
            fp = get_searchmap_filepath(item, self.filepath, binning=self.map_binnings['search'], pad_zeros=zero_padding)
            if fp.exists():
                return fp

    @staticmethod
    def _get_view_map_filepath_from_item(view_map_item, nav_filepath, pad_zeros=0, extension='mrc'):
        map_filepath = Path(view_map_item['MapFile'].replace("\\", '/'))
        map_section = get_map_sec_id(view_map_item) + 1
        map_section_str = ('{:0' + str(pad_zeros) + 'd}').format(map_section)
        return Path(nav_filepath).parent / f'{map_filepath.stem}_{map_section_str}.{extension}'

    def _get_view_map_filepath(self, item):
        for zero_padding in range(4):
            fp = self._get_view_map_filepath_from_item(item, self.filepath, pad_zeros=zero_padding)
            if fp.exists():
                return fp

    @staticmethod
    def _get_record_map_filepath_from_item(record_map_item, nav_filepath, binning=4, pad_zeros=0, extension='mrc'):
        map_filepath = Path(record_map_item['MapFile'].replace("\\", '/'))
        map_section = get_map_sec_id(record_map_item) + 1
        map_section_str = ('{:0' + str(pad_zeros) + 'd}').format(map_section)
        return Path(nav_filepath).parent / f'{map_filepath.stem}_{map_section_str}_bin{binning}.{extension}'

    def _get_record_map_filepath(self, item):
        for zero_padding in range(4):
            fp = self._get_record_map_filepath_from_item(item, self.filepath, binning=self.map_binnings['record'], pad_zeros=zero_padding)
            if fp.exists():
                return fp

    def _get_map_filepaths(self, map_type):
        if map_type == 'grid':
            return self._function_on_property(map_type, 'map_items_dict', self._get_grid_map_filepath)
        if map_type == 'search':
            return self._function_on_property(map_type, 'map_items_dict', self._get_search_map_filepath)
        if map_type == 'view':
            return self._function_on_property(map_type, 'map_items_dict', self._get_view_map_filepath)
        if map_type == 'record':
            return self._function_on_property(map_type, 'map_items_dict', self._get_record_map_filepath)
        return super().get_map_filepaths(map_type)

    def _get_grid_mdoc_filepath(self, item):
        fp = get_map_filepath_from_nav_item(self.filepath, item)
        return get_mdoc_filepath(fp)

    def _get_search_mdoc_filepath(self, item):
        fp = get_map_filepath_from_nav_item(self.filepath, item)
        return get_mdoc_filepath(fp)

    def _get_view_mdoc_filepath(self, fp):
        import re
        return Path(re.sub(r'_view(_\d+)?\.mrc$', r'_view.mrc.mdoc', str(fp)))

    def _get_record_mdoc_filepath(self, fp):
        import re
        return re.sub(r'_record(_\d+_bin\d+)?\.mrc$', r'_record.mrc.mdoc', str(fp))

    def _get_mdoc_filepaths(self, map_type):
        if map_type == 'grid':
            return self._function_on_property(map_type, 'map_items_dict', self._get_grid_mdoc_filepath)
        if map_type == 'search':
            return self._function_on_property(map_type, 'map_items_dict', self._get_search_mdoc_filepath)
        if map_type == 'view':
            return self._function_on_property(map_type, 'map_filepaths', self._get_view_mdoc_filepath)
        if map_type == 'record':
            return self._function_on_property(map_type, 'map_filepaths', self._get_record_mdoc_filepath)
        return super()._get_mdoc_filepaths(map_type)

    def _get_key_dependencies_for_map_type(self, map_type):

        if map_type == 'grid':
            return [list(self.map_items_dict[map_type].keys())[0]]

        if map_type == 'search':
            return [sorted(list(self.map_items_dict[map_type].keys()), key=self._sort_key)]

        if map_type == 'view':
            assignment_info = assign_view_maps_to_search_map(self.map_items_dict['view'], self.map_items_dict['search'], self.nav_dict['items'], return_items=False)
            return [[sorted(assignment_info[k], key=self._sort_key) for k in self.key_hierarchy['search'][0]]]

        if map_type == 'record':
            assignment_info = self._assign_record_maps_to_view_maps(self.map_items_dict['record'], self.map_items_dict['view'])
            return [[[[assignment_info[kkk]] for kkk in kk] for kk in k] for k in self.key_hierarchy['view']]


class TomoCLEMNavigator(Navigator):

    MAP_TYPES = [
        'grid',
        'mmm',
        'view',
        'tgt'
        # 'tilt_stack'
    ]

    SEARCH_STRINGS = dict(
        grid=None,
        mmm='MMM_*.mrc',
        view='*tgt_???_view.mrc',
        tgt='*tgt_???.mrc'
    )

    def __init__(
            self,
            filepath: str,
            grid_bin: int = 1,
            mmm_bin: int = 1,
            view_bin: int = 1,
            tgt_bin: int = 1
    ):
        super().__init__(
            filepath,
            map_types=self.MAP_TYPES,
            map_hierarchy=self.MAP_TYPES,
            search_strings=self.SEARCH_STRINGS,
            map_binnings=dict(
                grid=grid_bin,
                mmm=mmm_bin,
                view=view_bin,
                tgt=tgt_bin
            )
        )

        self.lamella_ids = {map_type: self._get_map_lamella_ids(map_type) for map_type in self.map_types}

    @staticmethod
    def _get_lamella_from_note(medium_mag_map_item):
        lamella_id = medium_mag_map_item['Note'].split(' - ')[2][:3]
        return lamella_id

    @staticmethod
    def _get_lamella_from_map_file(map_item):
        return map_item['MapFile'].split('\\')[-1][:3]

    def _get_map_item_by_lamella_id(self, map_items, lamella_id, return_key_only=False):
        if return_key_only:
            return [k for k, v in map_items.items() if self._get_lamella_from_map_file(v) == lamella_id]
        return {k: v for k, v in view_map_items.items() if
                v['MapFile'].split('\\')[-1][:3] == lamella_id}

    def _get_map_items_dict(self, map_type):

        if map_type == 'grid':
            key = next(iter(self.nav_dict['items']))
            map_items = {key: self.nav_dict['items'][key]}
            assert map_items[key]['MapFile'].endswith('.mrc')
            return map_items

        return super()._get_map_items_dict(map_type)

    def _get_grid_map_filepath(self, fp):
        binning = self.map_binnings['grid']
        return fp.parent / f'{fp.stem}_stitched_grid01_bin{binning}{fp.suffix}'

    def _get_mmm_map_filepath(self, fp):
        binning = self.map_binnings['mmm']
        return fp.parent / f'{fp.stem}_stitched_grid01_bin{binning}{fp.suffix}'

    def _get_view_map_filepath(self, fp):
        return self.filepath.parent.parent / 'pace' / fp.name

    def _get_tgt_map_filepath(self, fp):
        return self.filepath.parent.parent / 'pace' / fp.name

    def _get_map_filepath(self, map_type, key, item):
        fp = super()._get_map_filepath(item)
        if map_type == 'grid':
            return self._get_grid_map_filepath(fp)
        if map_type == 'mmm':
            return self._get_mmm_map_filepath(fp)
        if map_type == 'view':
            return self._get_view_map_filepath(fp)
        if map_type == 'tgt':
            return self._get_tgt_map_filepath(fp)
        return fp

    def _get_grid_mdoc_filepath(self, item):
        fp = get_map_filepath_from_nav_item(self.filepath, item)
        return get_mdoc_filepath(fp)

    def _get_mmm_mdoc_filepath(self, item):
        fp = get_map_filepath_from_nav_item(self.filepath, item)
        return get_mdoc_filepath(fp)

    def _get_view_mdoc_filepath(self, fp):
        import re
        return Path(re.sub(r'_view(_\d+)?\.mrc$', r'_view.mrc.mdoc', str(fp)))

    def _get_mdoc_filepaths(self, map_type):
        if map_type == 'grid':
            return self._function_on_property(map_type, 'map_items_dict', self._get_grid_mdoc_filepath)
        if map_type == 'mmm':
            return self._function_on_property(map_type, 'map_items_dict', self._get_mmm_mdoc_filepath)
        if map_type == 'view':
            pass
        return super()._get_mdoc_filepaths(map_type)

    @staticmethod
    def _get_map_sec_id(map_item):
        import re
        fp = map_item['MapFile']
        match = re.search(r'_([0-9]+)(?:_[^.]*)?\.mrc$', str(fp))
        if match:
            return int(match.group(1))
        return 0

    def _get_map_lamella_id(self, map_type, key, item):
        if map_type == 'mmm':
            return self._get_lamella_from_note(item)
        if map_type in ['view', 'tgt']:
            return self._get_lamella_from_map_file(item)
        return ''

    def _get_map_lamella_ids(self, map_type):
        return self._function_on_property(map_type, 'map_items_dict', self._get_map_lamella_id)

    def _assign_maps_to_mmm_maps(self, map_items):
        nav_dict_items = self.nav_dict['items']
        medium_mag_map_items = self.map_items_dict['mmm']
        lamella_ids = {k: self._get_lamella_from_note(v) for k, v in medium_mag_map_items.items()}

        return {
            k: self._get_map_item_by_lamella_id(map_items, v, return_key_only=True)
            for k, v in lamella_ids.items()
        }

    def _get_key_dependencies_for_map_type(self, map_type):

        if map_type == 'grid':
            return [list(self.map_items_dict[map_type].keys())[0]]

        if map_type == 'mmm':
            return [sorted(list(self.map_items_dict[map_type].keys()))]

        if map_type == 'view':
            assignment_info = self._assign_maps_to_mmm_maps(self.map_items_dict['view'])
            return [[sorted(assignment_info[k]) for k in self.key_hierarchy['mmm'][0]]]

        if map_type == 'tgt':
            assignment_info = self._assign_maps_to_mmm_maps(self.map_items_dict['tgt'])
            return [[[sorted(assignment_info[k])] for k in self.key_hierarchy['mmm'][0]]]
