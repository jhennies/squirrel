import numpy as np


def _load_data(path, key='data', pattern='*.tif'):
    import os
    if os.path.splitext(path)[1] == '.h5':
        from squirrel.library.io import load_h5_container
        return load_h5_container(path, key=key)
    from squirrel.library.io import load_tif_stack
    return load_tif_stack(path, pattern=pattern)


def _write_data(path, data, key='data'):
    import os
    if os.path.splitext(path)[1] == '.h5':
        from squirrel.library.io import write_h5_container
        write_h5_container(path, data, key=key)
        return
    from squirrel.library.io import write_tif_stack
    write_tif_stack(data, path)


def normalize_workflow(data):
    # FIXME implement for other data types
    assert data.dtype == 'uint8', f'Normalization only implemented for uint8 data, found {data.dtype}'
    # FIXME refactor and add tests

    # FIXME expose parameters
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

    # FIXME expose parameters
    def _apply_quantiles(image, quantiles=(0.1, 0.9), anchors=(0.2, 0.8)):
        qlow, qhigh = _get_quantiles(image, quantiles)
        alow, ahigh = np.array(anchors) * 255
        image = _normalize(image, alow, ahigh, qlow, qhigh)
        return image

    for idx, img in enumerate(data):
        data[idx] = _apply_quantiles(img)

    return data


def normalize_slices_workflow(
        in_path,
        out_path,
        pattern='*.tif',
        in_h5_key='data',
        out_h5_key='data',
        verbose=False
):

    if verbose:
        print(f'in_path = {in_path}')
        print(f'out_path = {out_path}')
        print(f'pattern = {pattern}')
        print(f'in_h5_key = {in_h5_key}')
        print(f'out_h5_key = {out_h5_key}')

    data = _load_data(in_path, key=in_h5_key, pattern=pattern)
    data = normalize_workflow(data)
    _write_data(out_path, data, key=out_h5_key)

