import math
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

    print('Creating link maps for gridmap ...')

    from squirrel.library.serial_em import get_search_map_on_grid_info
    search_map_names, _, transformed_stage_positions, grid_map_resolution, _ = get_search_map_on_grid_info(
        nav_filepath, grid_map_img_bin=grid_map_img_bin, verbose=verbose
    )

    # Get filepath of the gridmap image
    from squirrel.library.serial_em import get_gridmap_filepath
    gridmap_filepath = get_gridmap_filepath(nav_filepath)
    if verbose:
        print(f'gridmap_filepath = {gridmap_filepath}')
        print(f'transformed_stage_positions = {transformed_stage_positions}')
        print(f'search_map_names = {search_map_names}')
        print(f'grid_map_resolution_bin = {(grid_map_resolution * grid_map_img_bin)}')

    # Draw the final result
    from squirrel.library.image import draw_strings_on_image
    draw_strings_on_image(
        gridmap_filepath,
        os.path.join(out_dirpath, os.path.split(gridmap_filepath)[1]),
        strings=search_map_names,
        positions=transformed_stage_positions,  # + 674,
        font_size=30,
        verbose=verbose
    )


def _create_link_map_for_search_map(
        nav_filepath, search_map_id, search_map_item, view_map_items,
        out_dirpath,
        search_map_img_bin=4,
        pad_search_map_id=1,
        verbose=False
):

    print(f'search_map_id = {search_map_id}')

    from squirrel.library.serial_em import get_view_on_search_map_info
    view_map_names, _, transformed_stage_positions, search_map_resolution, _ = get_view_on_search_map_info(
        nav_filepath,
        search_map_id,
        search_map_item,
        view_map_items,
        search_map_img_bin=search_map_img_bin,
        verbose=verbose
    )

    # Get filepath of the search map image
    from squirrel.library.serial_em import get_searchmap_filepath
    searchmap_filepath = get_searchmap_filepath(search_map_item, nav_filepath, pad_zeros=pad_search_map_id)
    if verbose:
        print(f'searchmap_filepath = {searchmap_filepath}')
        print(f'searchmap_resolution_bin = {search_map_resolution * search_map_img_bin}')

    # Draw the final result
    from squirrel.library.image import draw_strings_on_image
    draw_strings_on_image(
        searchmap_filepath,
        os.path.join(out_dirpath, os.path.split(searchmap_filepath)[1]),
        strings=view_map_names,
        positions=transformed_stage_positions,
        font_size=30
    )


def _create_link_maps_for_search_maps(
        nav_filepath, out_dirpath, search_map_img_bin=4, verbose=False
):

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
        if view_map_ids:
            _create_link_map_for_search_map(
                nav_filepath,
                search_map_id,
                search_map_items[search_map_id],
                {k: v for k, v in view_map_items.items() if k in view_map_ids},
                out_dirpath,
                search_map_img_bin=search_map_img_bin,
                pad_search_map_id=int(math.log10(len(search_map_items))) + 1,
                verbose=verbose
            )
        else:
            print(f'Search map ID: {search_map_id} does not have corresponding views.')


def create_link_maps_workflow(
        nav_filepath,
        out_dirpath,
        verbose=False
):
    if verbose:
        print(f'nav_filepath = {nav_filepath}')
        print(f'out_dirpath = {out_dirpath}')

    if not os.path.exists(out_dirpath):
        os.mkdir(out_dirpath)

    _create_link_maps_for_gridmap(nav_filepath, out_dirpath, verbose=verbose)
    _create_link_maps_for_search_maps(nav_filepath, out_dirpath, verbose=verbose)


if __name__ == '__main__':
    # For development
    create_link_maps_workflow(
        '/media/julian/Data/projects/hennies/cryo_mobie_devel/grid03/nav_250226_grid03.nav',
        '/media/julian/Data/projects/hennies/cryo_mobie_devel/grid03-link-maps',
        verbose=True
    )

    create_link_maps_workflow(
        '/media/julian/Data/projects/hennies/cryo_mobie_devel/250521_test-maps/nav_250521_test-maps.nav',
        '/media/julian/Data/projects/hennies/cryo_mobie_devel/250521_test-maps-link-maps',
        verbose=True
    )

    # create_link_maps_workflow(
    #     '/mnt/icem/external/00_old-sessions/20250227_direct_sa0096_nelson_sf/screening_images/grid02/nav_250227_grid02.nav',
    #     '/media/julian/Data/tmp/nav_link_maps2/',
    #     verbose=True
    # )
    # create_link_maps_workflow(
    #     '/mnt/icem/external/00_old-sessions/20250227_direct_sa0096_nelson_sf/screening_images/grid04/nav_250227_grid04.nav',
    #     '/media/julian/Data/tmp/nav_link_maps3/',
    #     verbose=True
    # )
