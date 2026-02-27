
import numpy as np


def xcorr(
        fixed,
        moving,
        sigma=1.0,
        use_clahe=0,
        normalization=None,
        xcorr_func=None
):

    from vigra.filters import gaussianSmoothing
    xcorr_kwargs = dict()
    if xcorr_func is None:
        from skimage.registration import phase_cross_correlation
        xcorr_func = phase_cross_correlation()
        xcorr_kwargs = dict(
            upsample_factor=100,
            normalization=normalization
        )

    if use_clahe:
        if use_clahe == 1:
            use_clahe = 127
        from squirrel.library.normalization import clahe_on_image
        fixed_ = clahe_on_image(fixed, tile_grid_size=(use_clahe, use_clahe), tile_grid_in_pixels=True)
        moving_ = clahe_on_image(moving, tile_grid_size=(use_clahe, use_clahe), tile_grid_in_pixels=True)
    else:
        fixed_ = fixed.copy()
        moving_ = moving.copy()

    fixed_ = gaussianSmoothing(fixed_.astype('float32'), sigma)
    moving_ = gaussianSmoothing(moving_.astype('float32'), sigma)

    return xcorr_func(
        fixed_, moving_, **xcorr_kwargs
    )


def normalized_cross_correlation(fixed, moving, shift):
    """
    Compute the normalized cross-correlation (NCC) between a fixed image
    and a moving image shifted by `shift`.

    Parameters
    ----------
    fixed : ndarray
        Reference image.
    moving : ndarray
        Image to shift and compare.
    shift : array-like
        Shift to apply to `moving` (can be fractional). Same length as fixed.ndim.

    Returns
    -------
    ncc : float
        Normalized cross-correlation between shifted moving and fixed image.
        Range: [-1, 1], where 1 is perfect match.
    """
    from scipy.ndimage import shift as ndi_shift

    # Shift moving image
    shifted_moving = ndi_shift(moving, shift=shift, order=1, mode='nearest', prefilter=False)

    # Flatten images
    f = fixed.ravel()
    m = shifted_moving.ravel()

    # Subtract mean
    f_mean = f - np.mean(f)
    m_mean = m - np.mean(m)

    # Compute NCC
    ncc = np.sum(f_mean * m_mean) / np.sqrt(np.sum(f_mean ** 2) * np.sum(m_mean ** 2))

    return ncc


if __name__ == "__main__":
    pass
