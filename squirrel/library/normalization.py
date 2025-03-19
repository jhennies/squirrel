
import numpy as np


def _get_quantiles(array, quantiles=(0.1, 0.9), threshold=(1, 254), dilate_background=0):
    from copy import deepcopy
    this_array = deepcopy(array)
    if dilate_background:
        from vigra.filters import discErosion, gaussianSmoothing
        mask = 1 - discErosion((gaussianSmoothing(this_array, 1.0) > 1).astype('uint8'), dilate_background)
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
        quantiles=(0.1, 0.9),
        anchors=(0.2, 0.8),
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


def clahe_on_image(
        image,
        clip_limit=3.0,
        tile_grid_size=(127, 127),
        cast_dtype=None,
        invert_output=False,
        gaussian_sigma=0.0,
):
    from cv2 import createCLAHE
    clahe = createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    clahe_filtered = clahe.apply(image)

    if gaussian_sigma > 0.0:
        from vigra.filters import gaussianSmoothing
        clahe_filtered = gaussianSmoothing(clahe_filtered, gaussian_sigma)
    if invert_output:
        from squirrel.library.volume import invert_image
        clahe_filtered = invert_image(clahe_filtered)

    if cast_dtype is None:
        return clahe_filtered

    dtype_in = clahe_filtered.dtype
    dtype_out = np.dtype(cast_dtype)
    max_val_in = np.iinfo(dtype_in).max
    max_val_out = np.iinfo(dtype_out).max
    return (clahe_filtered.astype('float64') / max_val_in * max_val_out).astype(dtype_out)


def clahe_on_slices(
        stack,
        clip_limit=3.0,
        tile_grid_size=(127, 127),
        cast_dtype=None,
        invert_output=False,
        gaussian_sigma=0.0,
        z_range=None,
        n_workers=1
):

    from squirrel.library.data import norm_z_range
    stack_shape = stack.shape
    z_range = norm_z_range(z_range, stack_shape[0])

    if n_workers == 1:

        result_stack = []
        for idx in range(*z_range):
            img = stack[idx]
            result_stack.append(
                clahe_on_image(
                    img, clip_limit, tile_grid_size,
                    cast_dtype, invert_output, gaussian_sigma
                )
            )

    else:
        print(f'Running with {n_workers} CPUs')
        from multiprocessing import Pool

        with Pool(processes=n_workers) as p:
            tasks = []
            for idx in range(*z_range):
                img = stack[idx]
                tasks.append(
                    p.apply_async(
                        clahe_on_image, (
                            img, clip_limit, tile_grid_size,
                            cast_dtype, invert_output, gaussian_sigma
                        )
                    )
                )
            result_stack = [task.get() for task in tasks]

    return np.array(result_stack)
