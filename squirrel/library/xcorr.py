
def xcorr(fixed, moving, sigma=1.0):
    # from vigra.filters import gaussianGradientMagnitude
    from vigra.filters import gaussianSmoothing
    from skimage.registration import phase_cross_correlation

    # fixed_ = gaussianGradientMagnitude(fixed.astype('float32'), sigma)
    # moving_ = gaussianGradientMagnitude(moving.astype('float32'), sigma)
    fixed_ = gaussianSmoothing(fixed.astype('float32'), sigma)
    moving_ = gaussianSmoothing(moving.astype('float32'), sigma)

    return phase_cross_correlation(
        fixed_, moving_, upsample_factor=100, normalization=None
    )
