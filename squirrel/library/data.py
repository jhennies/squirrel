
import numpy as np


def invert_data(data):
    # FIXME add tests
    try:
        data = np.iinfo(data.dtype).max - data
    except ValueError:
        data = data.max() - data
    return data
