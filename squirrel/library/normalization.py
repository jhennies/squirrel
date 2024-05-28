
import numpy as np


def _get_quantiles(array, quantiles=(0.1, 0.9), threshold=(1, 254)):
    array_ = array[np.logical_and(array > threshold[0], array < threshold[1])]
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


def _apply_quantiles(image, quantiles=(0.1, 0.9), anchors=(0.2, 0.8)):
    qlow, qhigh = _get_quantiles(image, quantiles)
    alow, ahigh = np.array(anchors) * 255
    image = _normalize(image, alow, ahigh, qlow, qhigh)
    return image


def normalize_slices(
        stack,
        z_range=None,
        n_workers=1
):
    # FIXME implement for other data types
    assert stack.dtype == 'uint8', f'Normalization only implemented for uint8 data, found {stack.dtype}'

    from squirrel.library.data import norm_z_range
    stack_shape = stack.shape
    z_range = norm_z_range(z_range, stack_shape[0])

    if n_workers == 1:

        result_stack = []
        for idx in range(*z_range):
            img = stack[idx]
            result_stack.append(_apply_quantiles(img))

    else:
        print(f'Running with {n_workers} CPUs')
        from multiprocessing import Pool

        with Pool(processes=n_workers) as p:
            tasks = []
            for idx in range(*z_range):
                img = stack[idx]
                tasks.append(p.apply_async(_apply_quantiles, (img,)))
            result_stack = [task.get() for task in tasks]

    print(f'result_stack.shape = {result_stack.shape}')
    return np.array(result_stack)