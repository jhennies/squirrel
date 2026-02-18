
import numpy as np


def xcorr(
        fixed,
        moving,
        sigma=1.0,
        use_clahe=False
):

    from vigra.filters import gaussianSmoothing
    from skimage.registration import phase_cross_correlation

    if use_clahe:
        from squirrel.library.normalization import clahe_on_image
        fixed_ = clahe_on_image(fixed, tile_grid_size=(127, 127), tile_grid_in_pixels=True)
        moving_ = clahe_on_image(moving, tile_grid_size=(127, 127), tile_grid_in_pixels=True)

    fixed_ = gaussianSmoothing(fixed_.astype('float32'), sigma)
    moving_ = gaussianSmoothing(moving_.astype('float32'), sigma)

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

    Matches phase_cross_correlation(..., normalization=None)
    when max_shift >= max(image_shape).
    """

    import numpy as np
    from numpy.fft import fftn, ifftn
    from vigra.filters import gaussianSmoothing
    from skimage.registration._phase_cross_correlation import _upsampled_dft

    # --- preprocessing ---
    if use_clahe:
        from squirrel.library.normalization import clahe_on_image
        fixed = clahe_on_image(fixed, tile_grid_size=(127, 127), tile_grid_in_pixels=True)
        moving = clahe_on_image(moving, tile_grid_size=(127, 127), tile_grid_in_pixels=True)

    fixed = gaussianSmoothing(fixed.astype(np.float32), sigma)
    moving = gaussianSmoothing(moving.astype(np.float32), sigma)

    shape = fixed.shape
    midpoints = np.array([np.fix(dim / 2) for dim in shape])

    # --- FFT ---
    F = fftn(fixed)
    M = fftn(moving)

    # IMPORTANT: match normalization=None
    R = F * M.conj()

    cc = ifftn(R)
    cc_abs = np.abs(cc)

    # --- apply constraint only to integer search ---
    if max_shift < max(shape):
        # build wrapped coordinate grid like phase_cross_correlation
        coords = np.indices(shape)

        for d in range(len(shape)):
            coords[d] = np.where(coords[d] > midpoints[d],
                                 coords[d] - shape[d],
                                 coords[d])

        dist = np.sqrt(np.sum(coords**2, axis=0))
        mask = dist <= max_shift

        cc_abs = np.where(mask, cc_abs, -np.inf)

    # --- integer peak ---
    maxima = np.unravel_index(np.argmax(cc_abs), shape)
    maxima = np.array(maxima, dtype=np.float64)

    # wrap shifts
    shifts = maxima.copy()
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    # --- subpixel refinement (identical to skimage) ---
    if upsample_factor > 1:
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = 1
        dftshift = np.fix(upsampled_region_size * upsample_factor / 2.0)
        sample_region_offset = dftshift - shifts * upsample_factor

        cc_upsampled = _upsampled_dft(
            R.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset,
        )

        maxima = np.unravel_index(
            np.argmax(np.abs(cc_upsampled)),
            cc_upsampled.shape
        )
        maxima = np.array(maxima, dtype=np.float64)

        maxima -= dftshift
        shifts += maxima / upsample_factor

    return tuple(shifts), np.max(cc_abs)


if __name__ == "__main__":
    pass
