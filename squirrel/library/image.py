
import numpy as np


def get_bounds(image, return_ints=False):
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(image)
    # take the smallest points and use them as the top left of your crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)

    if return_ints:
        return np.array([top_left[0], top_left[1], bottom_right[0] + 1, bottom_right[1] + 1]).astype(float).tolist()
    else:
        # generate bounds
        bounds = np.s_[top_left[0]:bottom_right[0] + 1,  # plus 1 because slice isn't
                       top_left[1]:bottom_right[1] + 1]  # inclusive
        return bounds


def get_bounds_of_stack(stack_h, stack_shape, return_ints=False, z_range=None):

    from squirrel.library.io import load_data_from_handle_stack

    if z_range is None:
        z_range = [0, stack_shape[0]]

    return [
        get_bounds(load_data_from_handle_stack(stack_h, idx)[0], return_ints)
        for idx in range(*z_range)
    ]


def apply_auto_pad(transforms, stack_shape, stack_bounds, extra_padding=0, z_range=None):

    from squirrel.library.elastix import save_transforms
    from squirrel.library.transformation import validate_and_reshape_matrix
    transforms = np.array([validate_and_reshape_matrix(transform, ndim=2) for transform in transforms])

    def _transform_on_bounds(t, b):

        min_yx = [b[0], b[1]]
        min_y_max_x = [b[0], b[3]]
        max_y_min_x = [b[2], b[1]]
        max_yx = [b[2], b[3]]

        t_min_yx = np.matmul(np.linalg.inv(t), min_yx + [1.])
        t_min_y_max_x = np.matmul(np.linalg.inv(t), min_y_max_x + [1.])
        t_max_y_min_x = np.matmul(np.linalg.inv(t), max_y_min_x + [1.])
        t_max_yx = np.matmul(np.linalg.inv(t), max_yx + [1.])

        new_bounds = np.array([
            np.min([t_min_yx, t_min_y_max_x, t_max_y_min_x, t_max_yx], axis=0),
            np.max([t_min_yx, t_min_y_max_x, t_max_y_min_x, t_max_yx], axis=0)
        ])[:, :2]

        return new_bounds

    if z_range is None:
        z_range = [0, stack_shape[0]]

    new_bounds = []
    for idx in range(*z_range):
        t = transforms[idx]
        bounds_ = _transform_on_bounds(t, stack_bounds[idx])
        new_bounds.append(bounds_)

    new_bounds = np.array(new_bounds)
    new_bounds = [
        np.min(new_bounds[:, 0], axis=0),
        np.max(new_bounds[:, 1], axis=0)
    ]

    # Modify the offsets within the transforms to move everything towards the origin
    from squirrel.library.transformation import setup_translation_matrix
    for idx in range(*z_range):
        transform = transforms[idx]
        transforms[idx] = np.dot(
            transform,
            validate_and_reshape_matrix(setup_translation_matrix(new_bounds[0] - extra_padding, ndim=2), ndim=2)
        )

    transforms = transforms[:, :2]

    transforms = save_transforms(transforms, None, param_order='M', save_order='C', ndim=2)

    # Also modify the stack_shape now to crop or extend the images
    stack_shape[1:] = (new_bounds[1] - new_bounds[0] + 2 * extra_padding).tolist()

    return transforms, stack_shape


def image_to_shape(image, shape):

    image_shape = np.array(image.shape)
    shape = np.ceil(np.array(shape)).astype(int)

    max_shape = (
        max(image_shape[0], shape[0]),
        max(image_shape[1], shape[1])
    )

    new_image = np.zeros(max_shape, dtype=image.dtype)
    try:
        s = np.s_[
            :max_shape[0],
            :max_shape[1]
        ]
        new_image[s] = image[s]
        return new_image
    except ValueError:
        pass
    try:
        s = np.s_[
            :image_shape[0],
            :max_shape[1]
        ]
        new_image[s] = image[s]
        return new_image
    except ValueError:
        pass
    try:
        s = np.s_[
            :max_shape[0],
            :image_shape[1]
        ]
        new_image[s] = image[s]
        return new_image
    except ValueError:
        pass
    try:
        s = np.s_[
            :image_shape[0],
            :image_shape[1]
        ]
        new_image[s] = image[s]
        return new_image
    except ValueError:
        pass




