
import numpy as np


def pad_volume(vol, min_shape):

    if np.all(np.array(vol.shape) > np.array(min_shape)):
        return vol

    t_vol = np.zeros(np.max([vol.shape, min_shape], axis=0), dtype=vol.dtype)
    t_vol[:vol.shape[0], :vol.shape[1], :vol.shape[2]] = vol

    return t_vol
