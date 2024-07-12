
import numpy as np


def pad_volume(vol, min_shape, axes=None):

    if np.all(np.array(vol.shape) > np.array(min_shape)):
        return vol

    if axes is None:

        new_shape = np.max([vol.shape, min_shape], axis=0)

    else:
        new_shape = np.array(vol.shape)
        for a in axes:
            if vol.shape[a] < min_shape[a]:
                new_shape[a] = min_shape[a]

    t_vol = np.zeros(new_shape, dtype=vol.dtype)
    t_vol[:vol.shape[0], :vol.shape[1], :vol.shape[2]] = vol

    return t_vol

