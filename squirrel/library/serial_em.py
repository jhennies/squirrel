import numpy as np


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


# def get_rotation_matrix_from_pts(item_dict):
#
#     pts_x = get_value_list_from_item(item_dict, 'PtsX')
#     pts_y = get_value_list_from_item(item_dict, 'PtsY')
#
#     # Compute direction vector of the edge
#     dx, dy = np.array([pts_y[1], pts_x[1]]) - np.array([pts_y[0], pts_x[0]])
#
#     # Compute the rotation angle
#     theta = np.arctan2(dy, dx)  # Angle relative to x-axis
#
#     # Construct the rotation matrix
#     cos_theta, sin_theta = np.cos(-theta), np.sin(-theta)
#     rotation_matrix = np.array([[cos_theta, -sin_theta],
#                                 [sin_theta, cos_theta]])
#
#     # rotation_matrix = np.array([[1, 0],
#     #                            [0, 1]])
#
#     return rotation_matrix, theta  # Also returning the angle


# def compute_affine_transform(src_pts, dst_pts):
#     """
#     Computes the 2x3 affine transformation matrix that maps src_pts to dst_pts.
#
#     Parameters:
#     src_pts: List of 4 (x, y) points from the source rectangle (B).
#     dst_pts: List of 4 (x, y) points from the distorted shape (A).
# 0
#     Returns:
#     2x3 affine transformation matrix.
#     """
#     import cv2  # OpenCV for solving affine transformation
#     # Convert points to numpy array (needed for OpenCV)
#     src_pts = np.array(src_pts, dtype=np.float32)
#     dst_pts = np.array(dst_pts, dtype=np.float32)
#
#     # Compute the affine transformation matrix using OpenCV
#     affine_matrix = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])  # Uses only 3 points
#
#     return affine_matrix


# def get_affine_from_pts(item_dict, map_width_height):  # TODO get MapWidthHeight from navigator
#
#     pts_x = get_value_list_from_item(item_dict, 'PtsX')
#     pts_y = get_value_list_from_item(item_dict, 'PtsY')
#
#     half_mwh = np.array(map_width_height) / 2
#
#     # How the four corner points ...
#     src_pts = np.array([
#         [-half_mwh[0], -half_mwh[1]],
#         [-half_mwh[0], half_mwh[1]],
#         [half_mwh[0], half_mwh[1]],
#         [half_mwh[0], -half_mwh[1]]
#     ])
#     # src_pts = np.array([
#     #     [0, 0],
#     #     [0, map_width_height[1]],
#     #     [map_width_height[0], map_width_height[1]],
#     #     [map_width_height[0], 0]
#     # ])
#     # ... are transformed to form the final output orientation
#     dst_pts = np.array((pts_x, pts_y)).swapaxes(0, 1)[:4]
#
#     return compute_affine_transform(src_pts, dst_pts)
#     # return compute_affine_transform(dst_pts, src_pts)


def compute_perspective_transform(src_pts, dst_pts):
    """
    Computes the 3x3 perspective transformation matrix to map A (rectangle) to B (quadrilateral).
    """
    import cv2

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return perspective_matrix


def get_perspective_from_pts(item_dict, map_width_height):

    pts_x = get_value_list_from_item(item_dict, 'PtsX')
    pts_y = get_value_list_from_item(item_dict, 'PtsY')

    half_mwh = np.array(map_width_height) / 2

    # How the four corner points ...
    src_pts = np.array([
        [-half_mwh[0], -half_mwh[1]],
        [-half_mwh[0], half_mwh[1]],
        [half_mwh[0], half_mwh[1]],
        [half_mwh[0], -half_mwh[1]]
    ])
    # ... are transformed to form the final output orientation
    dst_pts = np.array((pts_x, pts_y)).swapaxes(0, 1)[:4]

    # return compute_perspective_transform(src_pts, dst_pts)
    return compute_perspective_transform(dst_pts, src_pts)


def apply_perspective_transform(points, perspective_matrix):
    """
    Applies the perspective transformation to a set of points.

    Parameters:
    points: List of (x, y) points to transform.
    perspective_matrix: 3x3 perspective transformation matrix.

    Returns:
    Transformed points as a NumPy array.
    """
    points = np.array(points)
    # Convert points to homogeneous coordinates (x, y) → (x, y, 1)
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # Shape: (N, 3)

    # Apply transformation
    transformed_points_homogeneous = np.dot(perspective_matrix, points_homogeneous.T).T  # Shape: (N, 3)

    # Convert back from homogeneous coordinates (divide by last column)
    transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2][:, np.newaxis]

    return transformed_points


def get_gridmap_filepath(nav_filepath):
    from glob import glob
    import os
    return glob(os.path.join(
        os.path.split(nav_filepath)[0],
        'gridmap_*.png'
    ))[0]


def get_searchmap_filepath(search_map_item, nav_filepath, binning=4):
    from glob import glob
    import os
    map_filepath = search_map_item['MapFile'].replace("\\", '/')
    map_section = int(search_map_item['MapSection']) + 1
    return os.path.join(
        os.path.split(nav_filepath)[0],
        f'{os.path.splitext(os.path.split(map_filepath)[1])[0]}_{map_section}_bin{binning}.png'
    )


def get_nav_item_id_from_note(in_item):
    return in_item['Note'].split(' ')[0]


def get_view_map_items_by_drawn_id(view_map_items, drawn_id, nav_dict_items):
    return [k for k, v in view_map_items.items() if nav_dict_items[get_nav_item_id_from_note(v)]['DrawnID'] == drawn_id]


def assign_view_maps_to_search_map(view_map_items, search_map_items, nav_dict_items):

    map_ids = {k: v['MapID'] for k, v in search_map_items.items()}
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


def get_resolution_of_nav_item(nav_item, mdoc_dirpath=None, unit='micrometer'):

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


