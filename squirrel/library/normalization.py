
import numpy as np


def _get_quantiles(array, quantiles=(0.1, 0.9), threshold=(1, 254), dilate_background=0):
    from copy import deepcopy
    this_array = deepcopy(array)
    if dilate_background:
        from vigra.filters import discErosion
        mask = 1 - discErosion((this_array > 1).astype('uint8'), dilate_background)
        this_array[mask > 0] = 0
    array_ = this_array[np.logical_and(this_array > threshold[0], this_array < threshold[1])]
    return np.quantile(array_, quantiles[0]), np.quantile(array_, quantiles[1])


def _normalize(pixels, alow, ahigh, qlow, qhigh):
    pixels = pixels.astype('float64')
    pixels -= qlow
    pixels /= qhigh - qlow
    pixels *= ahigh - alow
    pixels += alow
    pixels[pixels < 0] = 0
    pixels[pixels > 255] = 255
    pixels = np.round(pixels).astype('uint8')
    return pixels


def _apply_quantiles(image, quantiles=(0.1, 0.9), anchors=(0.2, 0.8), dilate_background=0):
    qlow, qhigh = _get_quantiles(image, quantiles, dilate_background=dilate_background)
    alow, ahigh = np.array(anchors) * 255
    image = _normalize(image, alow, ahigh, qlow, qhigh)
    return image


def normalize_slices(
        stack,
        dilate_background=0,
        z_range=None,
        n_workers=1
):
    # FIXME implement for other data types
    assert stack.dtype == 'uint8', f'Normalization only implemented for uint8 data, found {stack.dtype}'

    from squirrel.library.data import norm_z_range
    stack_shape = stack.shape
    z_range = norm_z_range(z_range, stack_shape[0])

    quantiles = (0.1, 0.9)
    anchors = (0.2, 0.8)

    if n_workers == 1:

        result_stack = []
        for idx in range(*z_range):
            img = stack[idx]
            result_stack.append(_apply_quantiles(img, quantiles, anchors, dilate_background))

    else:
        print(f'Running with {n_workers} CPUs')
        from multiprocessing import Pool

        with Pool(processes=n_workers) as p:
            tasks = []
            for idx in range(*z_range):
                img = stack[idx]
                tasks.append(p.apply_async(_apply_quantiles, (img, quantiles, anchors, dilate_background)))
            result_stack = [task.get() for task in tasks]

    return np.array(result_stack)