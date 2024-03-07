
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


def apply_auto_pad(transforms, stack_shape, stack_bounds, extra_padding=0):

    stack_bounds = np.array(stack_bounds)
    extra_padding = np.array([extra_padding] * 2)

    transforms = np.array(transforms)
    assert transforms.shape == (stack_shape[0], 6), \
        f'Currently only implemented for affine transforms of "C" layout; ' \
        f'transforms.shape={transforms.shape}; ' \
        f'stack_shape.shape={stack_shape}; '

    offsets = np.array([[affine[2], affine[5]] for affine in transforms])
    assert offsets.shape == (stack_shape[0], 2)

    # Modify the offsets within the transforms to move everything towards the origin
    starts = stack_bounds[:, :2]
    add_offset = -(extra_padding - (starts + offsets).min(axis=0))
    transforms[:, 2] = add_offset[0]
    transforms[:, 5] = add_offset[1]

    # Also modify the stack_shape now to crop or extend the images
    stops = stack_bounds[:, 2:]
    new_bounds_yx = (stops + offsets).max(axis=0) + 2 * extra_padding
    stack_shape[1:] = new_bounds_yx

    return transforms, stack_shape


def image_to_shape(image, shape):

    image_shape = np.array(image.shape)
    shape = np.ceil(np.array(shape)).astype(int)

    new_image = np.zeros(shape, dtype=image.dtype)
    s = np.s_[
        :min(image_shape[0], shape[0]),
        :min(image_shape[1], shape[1])
    ]
    new_image[s] = image[s]
    return new_image



