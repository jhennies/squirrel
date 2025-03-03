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


def create_link_maps_workflow(
        nav_filepath,
        out_dirpath,
        verbose=False
):
    if verbose:
        print(f'nav_filepath = {nav_filepath}')
        print(f'out_dirpath = {out_dirpath}')

    from squirrel.library.serial_em import (
        navigator_file_to_dict,
        get_map_items_by_map_file,
        get_raw_stage_xy_from_item,
        get_map_scale_matrix_from_item,
        get_value_list_from_item,
        get_rotation_matrix_from_pts,
        get_affine_from_pts,
        get_perspective_from_pts,
        apply_perspective_transform
    )
    nav_dict = navigator_file_to_dict(nav_filepath)
    search_map_items = get_map_items_by_map_file(nav_dict, '_search.mrc')
    grid_map_items = get_map_items_by_map_file(nav_dict, 'gridmap.st')
    # grid_map_scale_mat = get_map_scale_matrix_from_item(list(grid_map_items.values())[0])

    # grid_map_pts_x = get_value_list_from_item(list(grid_map_items.values())[0], 'PtsX')
    # grid_map_pts_y = get_value_list_from_item(list(grid_map_items.values())[0], 'PtsY')
    # grid_map_rot_matrix, _ = get_rotation_matrix_from_pts(list(grid_map_items.values())[0])
    map_width_height = get_value_list_from_item(list(grid_map_items.values())[0], 'MapWidthHeight')
    grid_map_affine_matrix = get_affine_from_pts(list(grid_map_items.values())[0], map_width_height)
    grid_map_perspective_transform = get_perspective_from_pts(list(grid_map_items.values())[0], np.array(map_width_height) * 0.2236)

    search_map_stage_positions = []
    search_map_names = []
    # search_map_scale_matrices = []
    # scaled_stage_positions = []
    # rotated_search_map_stage_positions = []
    # transformed_stage_positions = []
    for k, v in search_map_items.items():
        search_map_names.append(str(v['MapSection'] + 1))
        search_map_stage_positions.append(get_raw_stage_xy_from_item(v))
        # search_map_scale_matrices.append(get_map_scale_matrix_from_item(v))
        # scaled_stage_positions.append(np.dot(grid_map_scale_mat, get_raw_stage_xy_from_item(v)))
        # rotated_search_map_stage_positions.append(np.dot(grid_map_rot_matrix, get_raw_stage_xy_from_item(v)))
        # rotated_search_map_stage_positions.append(get_raw_stage_xy_from_item(v) @ grid_map_rot_matrix.T)
        # transformed_stage_positions.append(np.dot(get_raw_stage_xy_from_item(v) + [1], grid_map_affine_matrix.T))

    transformed_stage_positions = apply_perspective_transform(search_map_stage_positions, grid_map_perspective_transform)

    if verbose:
        print(f'search_map_names = {search_map_names}')
        print(f'search_map_stage_positions = {search_map_stage_positions}')
        # print(f'search_map_scale_matrices = {search_map_scale_matrices}')
        # print(f'scaled_stage_positions = {scaled_stage_positions}')
        # print(f'grid_map_scale_mat = {grid_map_scale_mat}')

    if not os.path.exists(out_dirpath):
        os.mkdir(out_dirpath)

    from glob import glob
    gridmap_filepath = glob(os.path.join(
        os.path.split(nav_filepath)[0],
        'gridmap_*.png'
    ))[0]

    from squirrel.library.image import draw_strings_on_image
    draw_strings_on_image(
        gridmap_filepath,
        os.path.join(out_dirpath, os.path.split(gridmap_filepath)[1]),
        strings=search_map_names,
        # positions=np.array(scaled_stage_positions) * (1/1.7888) * 0.2 + 674,  # (np.array(search_scmap_stage_positions)) * (1/1.7888) + 674,
        # positions=np.array(search_map_stage_positions) * (1/1.7888) + 674,  # (np.array(search_scmap_stage_positions)) * (1/1.7888) + 674,
        # positions=(np.array(rotated_search_map_stage_positions) + np.array([-10.11, 3.49])) * (1/1.7888) + 674,  # (np.array(search_scmap_stage_positions)) * (1/1.7888) + 674,
        positions=(np.array(transformed_stage_positions)[:, ::-1]) * (1/1.7888) + 674,
        font_size=30
    )


if __name__ == '__main__':
    # For development
    create_link_maps_workflow(
        '/mnt/icem/fromm/for_julian/grid03/nav_250226_grid03.nav',
        '/media/julian/Data/tmp/nav_link_maps/',
        verbose=True
    )
