
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


def scale_image_nearest(image, scale_factors):
    from scipy.ndimage import zoom
    return zoom(image, scale_factors, order=0)
