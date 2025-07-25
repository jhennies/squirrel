import numpy as np
from pathlib import Path

# ______________________________________________________________________________________________________________________
# Functions of the original linkmaps workflow

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


def get_map_sec_id(map_item):
    import re
    match = re.search(r'Sec (\d+)', map_item['Note'])
    return int(match.group(1)) if match else None


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

        # # map_type_relation defines the dependencies of the map types
        # # Example 1: {'grid': {'search': {'view': 'record'}}}
        # #   -> linear dependency where record depends on view, view on search and search on grid
        # # Example 2: {'grid': {'search': ['view', 'record']}}
        # #   -> view and record depend on search and search on grid
        # self.map_type_relation = self._get_map_type_relation()
        # self.map_type_relation_sequences = self._get_map_type_relation_sequence()
        # # item_relations is a dictionary that defines which item depends on which item from the higher order map_type
        # # Example 1: {'g1': {s1: {v1: [r1]}}}
        # # Example 2: {'g1': {s1: [[v1], [r1]]}}
        # self.item_relation = self._get_item_relation()
        # self.tree, self.flat_list = self._build_tree_and_flat_list()
        # pass

        self.map_hierarchy = None
        self.map_item_relation = None

    def _function_on_property(self, map_type, prop_name, func, **kwargs):
        return {k: func(v, **kwargs) for k, v in getattr(self, prop_name)[map_type].items()}

    def _get_map_type_relation_sequence(self):

        def build_path(key, mapping):
            path = []
            while key is not None:
                path.append(key)
                key = mapping.get(key)
            return path[::-1]

        return {k: build_path(k, self.map_type_relation) for k in self.map_type_relation}

    def _get_map_type_relation(self):
        # return self.map_types
        return dict(zip(self.map_types, [None] * len(self.map_types)))

    def _get_item_relation(self):
        return [list(self.map_items_dict[mt].keys()) for mt in self.map_types]

    def _get_map_items_dict(self, map_type):
        return get_map_items_by_glob(self.nav_dict, self.filepath, self.search_strings[map_type])

    def _get_map_filepath(self, item):
        return get_map_filepath_from_nav_item(self.filepath, item)

    def _get_map_filepaths(self, map_type):
        return self._function_on_property(map_type, 'map_items_dict', self._get_map_filepath)

    def _get_mdoc_filepaths(self, map_type):
        return self._function_on_property(map_type, 'map_filepaths', get_mdoc_filepath)

    def _get_map_resolution(self, item):
        fp = get_map_filepath_from_nav_item(self.filepath, item)
        mdoc_fp = get_mdoc_filepath(fp)
        return get_resolution_from_mdoc(mdoc_fp, unit='micrometer')

    def _get_affine(self, item, binning, invert=False, full_square=False, flatten=False):

        xy = get_map_scale_xy(item)
        mat = get_map_scale_matrix_from_item(item)
        img_shp = get_map_shape(item, binning)
        map_binning = get_value_from_item(item, 'MapBinning')
        mont_binning = get_value_from_item(item, 'MontBinning')
        map_resolution = self._get_map_resolution(item)

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

    def _get_map_full_affine(self, item, binning, apply_affine=None, invert=False, full_square=False, flatten=False):

        if apply_affine is not None:
            affine = self._get_affine(item, binning, invert=invert, full_square=full_square, flatten=False)
            if flatten:
                return (apply_affine @ affine).flatten()
            return apply_affine @ affine

        return self._get_affine(item, binning, invert=invert, full_square=full_square, flatten=flatten)

    def get_grid_key(self):
        return next(iter(self.map_items_dict['grid']))

    def get_grid_item(self):
        key = self.get_grid_key()
        return self.map_items_dict['grid'][key]

    def get_map_full_affines(self, map_type, stage_coordinate_system=False):

        kwargs = dict(
            apply_affine=(
                None if stage_coordinate_system else
                self._get_map_full_affine(self.get_grid_item(), self.map_binnings['grid'], invert=False, full_square=True, flatten=False)
            ),
            binning=self.map_binnings[map_type],
            invert=True, full_square=True, flatten=True
        )
        return self._function_on_property(map_type, 'map_items_dict', self._get_map_full_affine, **kwargs)

    def get_grid_affine(self, stage_coordinate_system=False):
        affines = self.get_map_full_affines('grid', stage_coordinate_system=stage_coordinate_system)
        return affines[next(iter(affines))]

    def _get_map_sec_ids(self, map_type):
        return self._function_on_property(map_type, 'map_items_dict', get_map_sec_id)

    def get_contrast_limits(self, map_type):
        if self.map_contrast_limits[map_type] is None:
            print(f'Computing contrast limits for {map_type}')
            self.map_contrast_limits[map_type] = self._function_on_property(map_type, 'map_filepaths', get_contrast_limits_from_map)
        return self.map_contrast_limits[map_type]

    def get_property(self, prop_name, map_type, map_ids=None, parent_map_id=None):
        if map_ids is not None:
            return [getattr(self, prop_name)[map_type][x] for x in map_ids]
        if parent_map_id is not None:
            parent_map_type = self.map_type_relation[map_type]
            map_ids = self.item_relation[parent_map_type][parent_map_id]
            return [getattr(self, prop_name)[map_type][x] for x in map_ids]
        return getattr(self, prop_name)[map_type]

    def get_function(self, func_name, map_type, map_ids=None, parent_map_id=None, **kwargs):
        if map_ids is not None:
            return [getattr(self, func_name)(map_type, **kwargs)[x] for x in map_ids]
        if parent_map_id is not None:
            parent_map_type = self.map_type_relation[map_type]
            map_ids = self.item_relation[parent_map_type][parent_map_id]
            return [getattr(self, func_name)(map_type, **kwargs)[x] for x in map_ids]
        return getattr(self, func_name)(map_type, **kwargs)

    def iterate_keys(self):
        pass

    def _build_tree_and_flat_list(
            hierarchy: Optional[List[str]],
            relation_functions: Dict[str, Callable[[], Dict[Any, Any]]]
    ) -> (List[Dict], List[Dict]):
        """
        hierarchy: list like ['grid', 'search', 'view', 'record'] or None
        relation_functions: dict mapping relation name (e.g., 'search_on_grid') to function returning mapping
        Returns: (tree, flat_list)
        """
        flat_list = []
        id_to_node = {}
        child_to_parent = {}

        # Step 1: Build flat list
        if hierarchy:
            for i in range(len(hierarchy) - 1):
                parent_type = hierarchy[i]
                child_type = hierarchy[i + 1]
                relation_key = f"{child_type}_on_{parent_type}"
                if relation_key not in relation_functions:
                    raise ValueError(f"Missing relation function for {relation_key}")

                relation = relation_functions[
                    relation_key]()  # returns {parent_id: [child_ids]} or {parent_id: child_id}

                for parent_id, children in relation.items():
                    if not isinstance(children, list):
                        children = [children]  # promote to list if one-to-one

                    for child_id in children:
                        flat_list.append({
                            "id": child_id,
                            "type": child_type,
                            "parent_id": parent_id
                        })
                        child_to_parent[child_id] = parent_id

        # Step 2: Add root-level items (if hierarchy exists)
        if hierarchy:
            root_type = hierarchy[0]
            roots_found = set(item["parent_id"] for item in flat_list if item["parent_id"] is not None)
            # Assume root items are keys in the first relation
            root_relation_key = f"{hierarchy[1]}_on_{hierarchy[0]}"
            root_relation = relation_functions[root_relation_key]()
            for root_id in root_relation.keys():
                flat_list.append({
                    "id": root_id,
                    "type": root_type,
                    "parent_id": None
                })
        else:
            # No hierarchy: flatten all items from all relation functions
            for key, func in relation_functions.items():
                mapping = func()
                for parent_id, children in mapping.items():
                    if not isinstance(children, list):
                        children = [children]
                    flat_list.append({"id": parent_id, "type": "unknown", "parent_id": None})
                    for child_id in children:
                        flat_list.append({"id": child_id, "type": "unknown", "parent_id": parent_id})

        # Step 3: Build tree from flat list
        tree = []
        id_to_node = {}

        for item in flat_list:
            node = {k: item[k] for k in ["id", "type", "parent_id"]}
            node["children"] = []
            id_to_node[node["id"]] = node

        for node in id_to_node.values():
            pid = node["parent_id"]
            if pid is None:
                tree.append(node)
            else:
                parent_node = id_to_node.get(pid)
                if parent_node:
                    parent_node["children"].append(node)

        return tree, flat_list


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
            search_strings=self.SEARCH_STRINGS,
            map_binnings=dict(
                record=record_bin,
                view=view_bin,
                search=search_bin,
                grid=grid_bin
            )
        )

    def _get_map_type_relation(self):
        mtr = super()._get_map_type_relation()
        mtr['search'] = 'grid'
        mtr['view'] = 'search'
        mtr['record'] = 'view'
        return mtr

    @staticmethod
    def assign_record_maps_to_view_maps(record_items, view_items):
        rec_maps_to_view_maps = dict()
        for view_k, view_v in view_items.items():
            vidx = view_v['Note'].split(' ')[0]
            for rec_k, rec_v in record_items.items():
                ridx = rec_v['Note'].split(' ')[0]
                if vidx == ridx:
                    rec_maps_to_view_maps[view_k] = rec_k
                    break
        return rec_maps_to_view_maps

    def _get_item_relation(self):
        assert len(self.map_items_dict['grid']) == 1, 'Only one grid item allowed!'

        view_on_search = assign_view_maps_to_search_map(
            self.map_items_dict['view'],
            self.map_items_dict['search'],
            self.nav_dict['items']
        )
        record_on_view = self.assign_record_maps_to_view_maps(
            self.map_items_dict['record'],
            self.map_items_dict['view']
        )
        # record_on_search = assign_view_maps_to_search_map(
        #     self.map_items_dict['record'],
        #     self.map_items_dict['search'],
        #     self.nav_dict['items']
        # )
        # view_rec_on_search = dict()
        # for k, v in view_on_search.items():
        #     view_rec_on_search[k] = [v, record_on_search[k]]

        # return {list(self.map_items_dict['grid'].keys())[0]: view_rec_on_search}

        item_relation = dict()
        # for map_type in self.map_type_relation_sequences['record']:
        item_relation['grid'] = {
            list(self.map_items_dict['grid'].keys())[0]: list(self.map_items_dict['search'].keys())
        }
        item_relation['search'] = view_on_search
        item_relation['view'] = record_on_view

        return item_relation

    def get_grid_map_filepath(self, item):
        fp = get_gridmap_filepath(self.filepath)
        return fp

    def get_search_map_filepath(self, item):
        for zero_padding in range(4):
            fp = get_searchmap_filepath(item, self.filepath, binning=self.map_binnings['search'], pad_zeros=zero_padding)
            if fp.exists():
                return fp

    @staticmethod
    def _get_view_map_filepath(view_map_item, nav_filepath, pad_zeros=0, extension='mrc'):
        map_filepath = Path(view_map_item['MapFile'].replace("\\", '/'))
        map_section = get_map_sec_id(view_map_item) + 1
        map_section_str = ('{:0' + str(pad_zeros) + 'd}').format(map_section)
        return Path(nav_filepath).parent / f'{map_filepath.stem}_{map_section_str}.{extension}'

    def get_view_map_filepath(self, item):
        for zero_padding in range(4):
            fp = self._get_view_map_filepath(item, self.filepath, pad_zeros=zero_padding)
            if fp.exists():
                return fp

    @staticmethod
    def _get_record_map_filepath(record_map_item, nav_filepath, binning=4, pad_zeros=0, extension='mrc'):
        map_filepath = Path(record_map_item['MapFile'].replace("\\", '/'))
        map_section = get_map_sec_id(record_map_item) + 1
        map_section_str = ('{:0' + str(pad_zeros) + 'd}').format(map_section)
        return Path(nav_filepath).parent / f'{map_filepath.stem}_{map_section_str}_bin{binning}.{extension}'

    def get_record_map_filepath(self, item):
        for zero_padding in range(4):
            fp = self._get_record_map_filepath(item, self.filepath, binning=self.map_binnings['record'], pad_zeros=zero_padding)
            if fp.exists():
                return fp

    def _get_map_filepaths(self, map_type):
        if map_type == 'grid':
            return self._function_on_property(map_type, 'map_items_dict', self.get_grid_map_filepath)
        if map_type == 'search':
            return self._function_on_property(map_type, 'map_items_dict', self.get_search_map_filepath)
        if map_type == 'view':
            return self._function_on_property(map_type, 'map_items_dict', self.get_view_map_filepath)
        if map_type == 'record':
            return self._function_on_property(map_type, 'map_items_dict', self.get_record_map_filepath)
        return super().get_map_filepaths(map_type)

