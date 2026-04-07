
import numpy as np


def create_nearest_position_mapping(input_length, scale):
    import math

    output_length = math.ceil(input_length * scale)
    return dict(
        zip(
            range(output_length),
            np.round(np.linspace(0, input_length - 1, output_length)).astype(int)
        )
    )


def scale_image(image, scale_factors, order=0):
    from scipy.ndimage import zoom
    return zoom(image, scale_factors, order=order)


def scale_image_nearest(image, scale_factors):
    return scale_image(image, scale_factors, order=0)


def average_bin_image(img, factor):
    dtype = img.dtype
    h, w = img.shape
    img = img[:h - h % factor, :w - w % factor]  # trim edges

    return img.reshape(
        h // factor, factor,
        w // factor, factor
    ).mean(axis=(1, 3)).astype(dtype)
