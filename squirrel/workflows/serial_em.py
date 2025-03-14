import os.path
import numpy as np


def parse_navigator_file_workflow(
        nav_filepath,
        output_filepath=None,
        verbose=False,
):

    if verbose:
        print(f'nav_filepath = {nav_filepath}')
        print(f'output_filepath = {output_filepath}')

    # Load navigator file
    # Parse the string to setup the dictionary
    from squirrel.library.serial_em import navigator_file_to_dict
    nav_dict = navigator_file_to_dict(nav_filepath)

    # Optionally write to file
    if output_filepath is not None:
        from squirrel.library.serial_em import navigator_dict_to_file
        navigator_dict_to_file(nav_dict, output_filepath)

    # Return result
    return nav_dict


def _create_link_maps_for_gridmap(
        nav_filepath, out_dirpath, grid_map_img_bin=8, verbose=False
):

    if verbose:
        print('Creating link maps for gridmap ...')

    # Load navigator
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

    # Get the search map information
    from squirrel.library.serial_em import get_raw_stage_xy_from_item
    search_map_names = []
    search_map_stage_positions = []
    for k, v in search_map_items.items():
        search_map_names.append(str(v['MapSection'] + 1))
        search_map_stage_positions.append(get_raw_stage_xy_from_item(v))

    # Compute the perspective transform
    from squirrel.library.serial_em import (
        apply_perspective_transform, get_perspective_from_pts, get_value_list_from_item
    )
    map_width_height = get_value_list_from_item(grid_map_items[next(iter(grid_map_items))], 'MapWidthHeight')
    grid_map_perspective_transform = get_perspective_from_pts(
        list(grid_map_items.values())[0], np.array(map_width_height) * grid_map_resolution  # 0.2236
    )
    transformed_stage_positions = apply_perspective_transform(search_map_stage_positions, grid_map_perspective_transform)

    # Get filepath of the gridmap image
    from squirrel.library.serial_em import get_gridmap_filepath
    gridmap_filepath = get_gridmap_filepath(nav_filepath)

    # Draw the final result
    from squirrel.library.image import draw_strings_on_image
    draw_strings_on_image(
        gridmap_filepath,
        os.path.join(out_dirpath, os.path.split(gridmap_filepath)[1]),
        strings=search_map_names,
        positions=(np.array(transformed_stage_positions)[:, ::-1]) * (1/(grid_map_resolution * grid_map_img_bin)),  # + 674,
        font_size=30
    )


def _create_link_map_for_search_map(
        nav_filepath, search_map_id, search_map_item, view_map_items,
        out_dirpath,
        search_map_img_bin=4
):

    # Compute the perspective transform
    from squirrel.library.serial_em import (
        apply_perspective_transform, get_perspective_from_pts, get_value_list_from_item
    )

    # Get resolution of view maps
    from squirrel.library.serial_em import get_resolution_of_nav_item
    search_map_resolution = get_resolution_of_nav_item(
        search_map_item,
        os.path.split(nav_filepath)[0],
        unit='micrometer'
    )

    # Get the view map information
    from squirrel.library.serial_em import get_raw_stage_xy_from_item
    view_map_names = []
    view_map_stage_positions = []
    for k, v in view_map_items.items():
        view_map_names.append(str(v['MapSection'] + 1))
        view_map_stage_positions.append(get_raw_stage_xy_from_item(v))

    # Compute the perspective transform
    from squirrel.library.serial_em import (
        apply_perspective_transform, get_perspective_from_pts, get_value_list_from_item
    )
    map_width_height = get_value_list_from_item(search_map_item, 'MapWidthHeight')
    search_map_perspective_transform = get_perspective_from_pts(
        search_map_item, np.array(map_width_height) * search_map_resolution
    )
    transformed_stage_positions = apply_perspective_transform(view_map_stage_positions, search_map_perspective_transform)

    # Get filepath of the search map image
    from squirrel.library.serial_em import get_searchmap_filepath
    searchmap_filepath = get_searchmap_filepath(search_map_item, nav_filepath)

    # Draw the final result
    from squirrel.library.image import draw_strings_on_image
    draw_strings_on_image(
        searchmap_filepath,
        os.path.join(out_dirpath, os.path.split(searchmap_filepath)[1]),
        strings=view_map_names,
        positions=(np.array(transformed_stage_positions)[:, ::-1]) * (1/(search_map_resolution * search_map_img_bin)),
        font_size=30
    )


def _create_link_maps_for_search_maps(
        nav_filepath, out_dirpath, search_map_img_bin=4, verbose=False
):
    if verbose:
        print('Creating link maps for search map ...')

    # Load navigator
    from squirrel.library.serial_em import navigator_file_to_dict
    nav_dict = navigator_file_to_dict(nav_filepath)

    # Get relevant navigator items
    from squirrel.library.serial_em import get_map_items_by_map_file
    search_map_items = get_map_items_by_map_file(nav_dict, '_search.mrc')
    view_map_items = get_map_items_by_map_file(nav_dict, '_view.mrc')

    # Figure out which view maps belong to which search map
    from squirrel.library.serial_em import assign_view_maps_to_search_map
    view_to_search_map_items = assign_view_maps_to_search_map(view_map_items, search_map_items, nav_dict['items'])

    for search_map_id, view_map_ids in view_to_search_map_items.items():
        _create_link_map_for_search_map(
            nav_filepath,
            search_map_id,
            search_map_items[search_map_id],
            {k: v for k, v in view_map_items.items() if k in view_map_ids},
            out_dirpath,
            search_map_img_bin=search_map_img_bin
        )


def create_link_maps_workflow(
        nav_filepath,
        out_dirpath,
        verbose=False
):
    if verbose:
        print(f'nav_filepath = {nav_filepath}')
        print(f'out_dirpath = {out_dirpath}')

    _create_link_maps_for_gridmap(nav_filepath, out_dirpath, verbose=verbose)
    _create_link_maps_for_search_maps(nav_filepath, out_dirpath, verbose=verbose)


if __name__ == '__main__':
    # For development
    create_link_maps_workflow(
        '/mnt/icem/fromm/for_julian/grid03/nav_250226_grid03.nav',
        '/media/julian/Data/tmp/nav_link_maps/',
        verbose=True
    )
