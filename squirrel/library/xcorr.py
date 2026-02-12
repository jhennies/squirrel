
import numpy as np


def xcorr(fixed, moving, sigma=1.0):

    from vigra.filters import gaussianSmoothing
    from skimage.registration import phase_cross_correlation

    fixed_ = gaussianSmoothing(fixed.astype('float32'), sigma)
    moving_ = gaussianSmoothing(moving.astype('float32'), sigma)

    return phase_cross_correlation(
        fixed_, moving_, upsample_factor=100, normalization=None
    )


def xcorr_limited(
        fixed,
        moving,
        sigma=1.0,
        max_shift=10,
        upsample_factor=100,
        use_clahe=False
):
    """
    Phase cross-correlation with:
        - Gaussian smoothing
        - Euclidean shift constraint
        - Subpixel refinement

    Returns
    -------
    shift : tuple (row_shift, col_shift)  (subpixel accurate)
    peak_value : float
    """

    from numpy.fft import fftn, ifftn, fftshift
    from vigra.filters import gaussianSmoothing
    from skimage.registration._phase_cross_correlation import _upsampled_dft

    # --- pre-processing ---
    if use_clahe:
        from squirrel.library.normalization import clahe_on_image
        fixed = clahe_on_image(fixed, tile_grid_size=(127, 127), tile_grid_in_pixels=True)
        moving = clahe_on_image(moving, tile_grid_size=(127, 127), tile_grid_in_pixels=True)
    fixed = gaussianSmoothing(fixed.astype(np.float32), sigma)
    moving = gaussianSmoothing(moving.astype(np.float32), sigma)

    shape = fixed.shape
    center = np.array(shape) // 2

    # --- Fourier transforms ---
    F = fftn(fixed)
    M = fftn(moving)

    R = F * M.conj()
    R /= np.maximum(np.abs(R), 1e-12)  # phase-only normalization

    cc = np.real(ifftn(R))
    cc = fftshift(cc)

    # --- build radius mask ---
    y, x = np.indices(shape)
    dy = y - center[0]
    dx = x - center[1]
    dist = np.sqrt(dy**2 + dx**2)

    mask = dist <= max_shift
    cc_masked = np.where(mask, cc, -np.inf)

    # --- find best integer peak inside constraint ---
    maxpos = np.unravel_index(np.argmax(cc_masked), shape)
    integer_shift = np.array(maxpos) - center

    # --- Subpixel refinement using upsampled DFT ---
    # region around detected peak
    upsample_region_size = 3  # small local region

    # location in Fourier space
    sample_region_offset = integer_shift * upsample_factor

    cc_upsampled = _upsampled_dft(
        R.conj(),
        upsample_region_size,
        upsample_factor,
        sample_region_offset
    )

    maxima = np.unravel_index(np.argmax(np.abs(cc_upsampled)),
                              cc_upsampled.shape)

    # subpixel correction
    maxima = np.array(maxima) - upsample_region_size // 2
    subpixel_shift = integer_shift + maxima / upsample_factor

    # Final safety check (guarantee constraint)
    if np.linalg.norm(subpixel_shift) > max_shift:
        subpixel_shift = (
            subpixel_shift / np.linalg.norm(subpixel_shift)
        ) * max_shift

    return tuple(subpixel_shift), cc[maxpos]


if __name__ == "__main__":
    pass
