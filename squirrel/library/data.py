
import numpy as np


def invert_data(data):
    # FIXME add tests
    try:
        data = np.iinfo(data.dtype).max - data
    except ValueError:
        data = data.max() - data
    return data


def resolution_to_pixels(value, resolution):
    if type(value) in [list, tuple, np.array]:
        return np.array(value) / resolution
    return value / resolution
