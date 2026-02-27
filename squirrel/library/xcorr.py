
import numpy as np


def xcorr(
        fixed,
        moving,
        sigma=1.0,
        use_clahe=0,
        normalization=None
):

    from vigra.filters import gaussianSmoothing
    from skimage.registration import phase_cross_correlation

    if use_clahe:
        if use_clahe == 1:
            use_clahe = 127
        from squirrel.library.normalization import clahe_on_image
        fixed_ = clahe_on_image(fixed, tile_grid_size=(use_clahe, use_clahe), tile_grid_in_pixels=True, auto_mask=True)
        moving_ = clahe_on_image(moving, tile_grid_size=(use_clahe, use_clahe), tile_grid_in_pixels=True, auto_mask=True)
    else:
        fixed_ = fixed.copy()
        moving_ = moving.copy()

    fixed_ = gaussianSmoothing(fixed_.astype('float32'), sigma)
    moving_ = gaussianSmoothing(moving_.astype('float32'), sigma)

    shift, _, _ = phase_cross_correlation(
        fixed_, moving_, upsample_factor=100, normalization=normalization
    )

    return shift, 1 - normalized_cross_correlation(fixed_, moving_, shift)


def normalized_cross_correlation(fixed, moving, shift, erode_mask=0):
    """
    Compute normalized cross-correlation (NCC) ignoring zero-value pixels
    in either fixed or moving image.
    """
    from scipy.ndimage import shift as ndi_shift

    # Shift moving image
    shifted_moving = ndi_shift(moving, shift=shift, order=1, prefilter=False)

    from matplotlib import pyplot as plt
    plt.imshow(shifted_moving)
    plt.figure()
    plt.imshow(fixed)

    # Create mask: only pixels non-zero in both images
    mask = (fixed != 0) & (shifted_moving != 0)
    if erode_mask:
        from scipy.ndimage import binary_erosion
        mask = binary_erosion(mask, structure=None, iterations=erode_mask)

    plt.figure()
    plt.imshow(mask)
    plt.show()

    if np.count_nonzero(mask) == 0:
        raise ValueError("No overlapping non-zero pixels for NCC calculation")

    f_valid = fixed[mask]
    m_valid = shifted_moving[mask]

    # Subtract mean
    f_mean = f_valid - np.mean(f_valid)
    m_mean = m_valid - np.mean(m_valid)

    # Compute NCC
    ncc = np.sum(f_mean * m_mean) / np.sqrt(np.sum(f_mean**2) * np.sum(m_mean**2))

    return ncc


if __name__ == "__main__":
    from tifffile import imread
    a = imread('/media/julian/Data/tmp/xcorr-test/fixed2.tif')
    b = imread('/media/julian/Data/tmp/xcorr-test/moving2.tif')
    shft, err = xcorr(a, b, sigma=4, use_clahe=127)

    print(shft)
    print(err)
