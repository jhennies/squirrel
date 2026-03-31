
import numpy as np


def _compute_psr(CC, exclusion_radius=5):
    """
    Compute shift using phase_cross_correlation and PSR for ROI.
    """

    # Find peak
    peak_idx = np.unravel_index(np.argmax(CC), CC.shape)
    peak_value = CC[peak_idx]

    # Mask out a small neighborhood around the peak
    mask = np.ones_like(CC, dtype=bool)
    y, x = np.ogrid[:CC.shape[0], :CC.shape[1]]
    mask_area = (y - peak_idx[0]) ** 2 + (x - peak_idx[1]) ** 2 <= exclusion_radius ** 2
    mask[mask_area] = False

    bg = CC[mask]
    psr = (peak_value - bg.mean()) / (bg.std() + 1e-8)

    return psr, peak_value


def _peak_sharpness(CC, peak_idx, window=3):
    """
    Approximate sharpness by second derivative around the peak
    """
    from scipy.ndimage import gaussian_filter
    y, x = peak_idx

    # cast to int
    y = int(round(y))
    x = int(round(x))

    y0 = max(0, y - window)
    y1 = min(CC.shape[0], y + window + 1)
    x0 = max(0, x - window)
    x1 = min(CC.shape[1], x + window + 1)

    patch = CC[y0:y1, x0:x1]
    patch_smooth = gaussian_filter(patch, sigma=1)

    # Laplacian as a simple measure of curvature
    try:
        laplacian = np.sum(np.abs(np.gradient(np.gradient(patch_smooth)[0])[0] +
                                  np.abs(np.gradient(np.gradient(patch_smooth)[1])[1])))
    except ValueError:
        return 0
    return laplacian


def phase_cross_correlation(
    reference_image,
    moving_image,
    *,
    upsample_factor=1,
    space="real",
    disambiguate=False,
    reference_mask=None,
    moving_mask=None,
    overlap_ratio=0.3,
    normalization="phase",
):
    """Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT [1]_.

    Parameters
    ----------
    reference_image : array
        Reference image.
    moving_image : array
        Image to register. Must be same dimensionality as
        ``reference_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel. Default is 1 (no upsampling).
        Not used if any of ``reference_mask`` or ``moving_mask`` is not None.
    space : string, one of "real" or "fourier", optional
        Defines how the algorithm interprets input data. "real" means
        data will be FFT'd to compute the correlation, while "fourier"
        data will bypass FFT of input data. Case insensitive. Not
        used if any of ``reference_mask`` or ``moving_mask`` is not
        None.
    disambiguate : bool
        The shift returned by this function is only accurate *modulo* the
        image shape, due to the periodic nature of the Fourier transform. If
        this parameter is set to ``True``, the *real* space cross-correlation
        is computed for each possible shift, and the shift with the highest
        cross-correlation within the overlapping area is returned.
    reference_mask : ndarray
        Boolean mask for ``reference_image``. The mask should evaluate
        to ``True`` (or 1) on valid pixels. ``reference_mask`` should
        have the same shape as ``reference_image``.
    moving_mask : ndarray or None, optional
        Boolean mask for ``moving_image``. The mask should evaluate to ``True``
        (or 1) on valid pixels. ``moving_mask`` should have the same shape
        as ``moving_image``. If ``None``, ``reference_mask`` will be used.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for
        translations corresponding with an overlap ratio lower than this
        threshold will be ignored. A lower `overlap_ratio` leads to smaller
        maximum translation, while a higher `overlap_ratio` leads to greater
        robustness against spurious matches due to small overlap between
        masked images. Used only if one of ``reference_mask`` or
        ``moving_mask`` is not None.
    normalization : {"phase", None}
        The type of normalization to apply to the cross-correlation. This
        parameter is unused when masks (`reference_mask` and `moving_mask`) are
        supplied.

    Returns
    -------
    shift : ndarray
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``. Axis ordering is consistent with
        the axis order of the input array.
    error : float
        Translation invariant normalized RMS error between
        ``reference_image`` and ``moving_image``. For masked cross-correlation
        this error is not available and NaN is returned.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative). For masked cross-correlation
        this phase difference is not available and NaN is returned.

    Notes
    -----
    The use of cross-correlation to estimate image translation has a long
    history dating back to at least [2]_. The "phase correlation"
    method (selected by ``normalization="phase"``) was first proposed in [3]_.
    Publications [1]_ and [2]_ use an unnormalized cross-correlation
    (``normalization=None``). Which form of normalization is better is
    application-dependent. For example, the phase correlation method works
    well in registering images under different illumination, but is not very
    robust to noise. In a high noise scenario, the unnormalized method may be
    preferable.

    When masks are provided, a masked normalized cross-correlation algorithm is
    used [5]_, [6]_.

    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`
    .. [2] P. Anuta, Spatial registration of multispectral and multitemporal
           digital imagery using fast Fourier transform techniques, IEEE Trans.
           Geosci. Electron., vol. 8, no. 4, pp. 353–368, Oct. 1970.
           :DOI:`10.1109/TGE.1970.271435`.
    .. [3] C. D. Kuglin D. C. Hines. The phase correlation image alignment
           method, Proceeding of IEEE International Conference on Cybernetics
           and Society, pp. 163-165, New York, NY, USA, 1975, pp. 163–165.
    .. [4] James R. Fienup, "Invariant error metrics for image reconstruction"
           Optics Letters 36, 8352-8357 (1997). :DOI:`10.1364/AO.36.008352`
    .. [5] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [6] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`
    """

    from skimage.registration._phase_cross_correlation import (
        _masked_phase_cross_correlation, _upsampled_dft, _disambiguate_shift, _compute_error, _compute_phasediff
    )
    from scipy.fft import fftn, ifftn
    if (reference_mask is not None) or (moving_mask is not None):
        shift = _masked_phase_cross_correlation(
            reference_image, moving_image, reference_mask, moving_mask, overlap_ratio
        )
        return shift, np.nan, np.nan, np.nan

    # images must be the same shape
    if reference_image.shape != moving_image.shape:
        raise ValueError("images must be same shape")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = reference_image
        target_freq = moving_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq = fftn(reference_image)
        target_freq = fftn(moving_image)
    else:
        raise ValueError('space argument must be "real" of "fourier"')

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if normalization == "phase":
        eps = np.finfo(image_product.real.dtype).eps
        image_product /= np.maximum(np.abs(image_product), 100 * eps)
    elif normalization is not None:
        raise ValueError("normalization must be either phase or None")
    cross_correlation = ifftn(image_product)
    CC_real = np.abs(cross_correlation)

    # Locate maximum
    maxima = np.unravel_index(
        np.argmax(np.abs(cross_correlation)), cross_correlation.shape
    )
    midpoint = np.array([np.fix(axis_size / 2) for axis_size in shape])

    float_dtype = image_product.real.dtype

    shift = np.stack(maxima).astype(float_dtype, copy=False)
    shift[shift > midpoint] -= np.array(shape)[shift > midpoint]

    if upsample_factor == 1:
        src_amp = np.sum(np.real(src_freq * src_freq.conj()))
        src_amp /= src_freq.size
        target_amp = np.sum(np.real(target_freq * target_freq.conj()))
        target_amp /= target_freq.size
        CCmax = cross_correlation[maxima]
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        upsample_factor = np.array(upsample_factor, dtype=float_dtype)
        shift = np.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shift * upsample_factor
        cross_correlation = _upsampled_dft(
            image_product.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset,
        ).conj()
        # Locate maximum and map back to original pixel grid
        maxima = np.unravel_index(
            np.argmax(np.abs(cross_correlation)), cross_correlation.shape
        )
        CCmax = cross_correlation[maxima]

        maxima = np.stack(maxima).astype(float_dtype, copy=False)
        maxima -= dftshift

        shift += maxima / upsample_factor

        src_amp = np.sum(np.real(src_freq * src_freq.conj()))
        target_amp = np.sum(np.real(target_freq * target_freq.conj()))

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shift[dim] = 0

    if disambiguate:
        if space.lower() != 'real':
            reference_image = ifftn(reference_image)
            moving_image = ifftn(moving_image)
        shift = _disambiguate_shift(reference_image, moving_image, shift)

    # Redirect user to masked_phase_cross_correlation if NaNs are observed
    if np.isnan(CCmax) or np.isnan(src_amp) or np.isnan(target_amp):
        raise ValueError(
            "NaN values found, please remove NaNs from your "
            "input data or use the `reference_mask`/`moving_mask` "
            "keywords, eg: "
            "phase_cross_correlation(reference_image, moving_image, "
            "reference_mask=~np.isnan(reference_image), "
            "moving_mask=~np.isnan(moving_image))"
        )

    psr, peak_value = _compute_psr(CC_real)
    sharpness = _peak_sharpness(CC_real, maxima)

    return shift, psr, peak_value, sharpness
    # return shift, _compute_error(CCmax, src_amp, target_amp), _compute_phasediff(CCmax)


def xcorr(
        fixed,
        moving,
        sigma=1.0,
        use_clahe=0,
        normalization=None
):

    from vigra.filters import gaussianSmoothing
    # from skimage.registration import phase_cross_correlation

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

    shift, psr, peak_value, sharpness = phase_cross_correlation(
        fixed_, moving_, upsample_factor=100, normalization=normalization
    )

    return shift, 1 - normalized_cross_correlation(fixed_, moving_, shift), psr, peak_value, sharpness


def normalized_cross_correlation(fixed, moving, shift, erode_mask=0):
    """
    Compute normalized cross-correlation (NCC) ignoring zero-value pixels
    in either fixed or moving image.
    """
    from scipy.ndimage import shift as ndi_shift

    # Shift moving image
    shifted_moving = ndi_shift(moving, shift=shift, order=1, prefilter=False)

    # from matplotlib import pyplot as plt
    # plt.imshow(shifted_moving)
    # plt.figure()
    # plt.imshow(fixed)

    # Create mask: only pixels non-zero in both images
    mask = (fixed != 0) & (shifted_moving != 0)
    if erode_mask:
        from scipy.ndimage import binary_erosion
        mask = binary_erosion(mask, structure=None, iterations=erode_mask)

    # plt.figure()
    # plt.imshow(mask)
    # plt.show()

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


# def windowed_ncc(template, image, window_size=32):
#     """
#     Compute windowed normalized cross-correlation between template and image.
#
#     Args:
#         template: 2D array (ROI from slice i)
#         image: 2D array (ROI from slice i+1)
#         window_size: size of local window for normalization
#
#     Returns:
#         ncc_map: normalized cross-correlation map
#     """
#     from scipy.signal import correlate2d
#     from scipy.ndimage import gaussian_filter
#
#     # Convert to float
#     template = template.astype(np.float32)
#     image = image.astype(np.float32)
#
#     # Compute local mean and std for template
#     template_mean = gaussian_filter(template, window_size / 4)
#     template_std = np.sqrt(gaussian_filter((template - template_mean) ** 2, window_size / 4) + 1e-8)
#
#     # Compute local mean and std for image
#     image_mean = gaussian_filter(image, window_size / 4)
#     image_std = np.sqrt(gaussian_filter((image - image_mean) ** 2, window_size / 4) + 1e-8)
#
#     # Compute numerator (cross-correlation)
#     numerator = correlate2d(image - image_mean, template - template_mean, mode='same')
#
#     # Denominator (normalization)
#     denominator = template_std.sum() * image_std
#
#     ncc_map = numerator / (denominator + 1e-8)
#     return ncc_map
#
#
# def windowed_ncc_registration(fixed, moving, window_size=32, use_clahe=0, sigma=0.0):
#
#     from vigra.filters import gaussianSmoothing
#     from skimage.registration import phase_cross_correlation
#
#     if use_clahe:
#         if use_clahe == 1:
#             use_clahe = 127
#         from squirrel.library.normalization import clahe_on_image
#         fixed_ = clahe_on_image(fixed, tile_grid_size=(use_clahe, use_clahe), tile_grid_in_pixels=True, auto_mask=True)
#         moving_ = clahe_on_image(moving, tile_grid_size=(use_clahe, use_clahe), tile_grid_in_pixels=True, auto_mask=True)
#     else:
#         fixed_ = fixed.copy()
#         moving_ = moving.copy()
#
#     fixed_ = gaussianSmoothing(fixed_.astype('float32'), sigma)
#     moving_ = gaussianSmoothing(moving_.astype('float32'), sigma)
#
#     # Compute NCC map
#     ncc_map = windowed_ncc(fixed_, moving_, window_size=window_size)
#
#     # Find peak (translation)
#     max_idx = np.unravel_index(np.argmax(ncc_map), ncc_map.shape)
#     dy, dx = max_idx[0] - fixed_.shape[0]//2, max_idx[1] - fixed_.shape[1]//2
#
#     shift = [dy, dx]
#
#     return shift, 1 - normalized_cross_correlation(fixed, moving, shift)


if __name__ == "__main__":
    from tifffile import imread
    a = imread('/media/julian/Data/tmp/xcorr-test/fixed2.tif')
    b = imread('/media/julian/Data/tmp/xcorr-test/moving2.tif')
    shft, err = xcorr(a, b, sigma=4, use_clahe=127)

    print(shft)
    print(err)
