import os

import numpy as np


def apply_transforms_on_image(
        image,
        transforms,
        n_workers=os.cpu_count(),
        verbose=False
):

    if type(image) == str:
        from tifffile import imread
        image = imread(image)

    import SimpleITK as sitk
    from squirrel.library.affine_matrices import AffineMatrix

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(n_workers)

    txif = sitk.TransformixImageFilter()
    txif.LogToConsoleOff()
    if verbose:
        txif.LogToConsoleOn()
    for transform in transforms[::-1]:
        if type(transform) == AffineMatrix:
            transform = transform.to_elastix_affine(shape=image.shape, return_parameter_map=True)
        txif.AddTransformParameterMap(transform)
    txif.SetMovingImage(sitk.GetImageFromArray(image))
    txif.Execute()
    result_final = sitk.GetArrayFromImage(txif.GetResultImage())
    iinfo = np.iinfo(image.dtype)
    return np.clip(result_final, iinfo.min, iinfo.max).astype(image.dtype)


def apply_transforms_on_image_stack_slice(
        image_stack_h,
        image_idx,
        transforms,
        target_image_shape=None,
        n_slices=None,
        n_workers=os.cpu_count(),
        key=None,
        pattern=None,
        quiet=False,
        verbose=False
):

    if type(image_stack_h) == str:
        from squirrel.library.io import load_data_handle
        image_stack_h, _ = load_data_handle(image_stack_h, key=key, pattern=pattern)

    if type(transforms[0]) == str:
        import SimpleITK as sitk
        for idx, transform in enumerate(transforms):
            transforms[idx] = sitk.ReadParameterFile(transform)

    target_image_shape = image_stack_h[0].shape if target_image_shape is None else target_image_shape
    if not quiet or verbose:
        print(f'image_idx = {image_idx} / {n_slices}')

    from squirrel.library.io import get_reshaped_data
    z_slice = get_reshaped_data(image_stack_h, image_idx, target_image_shape)

    return apply_transforms_on_image(
        z_slice,
        transforms,
        n_workers=n_workers
    )[:target_image_shape[0], :target_image_shape[1]]

# def save_transforms(parameters, out_filepath, param_order='M', save_order='M', ndim=3, verbose=False):
#
#     parameters = np.array(parameters)
#     if verbose:
#         print(f'parameters = {parameters}')
#
#     def _elastix2m(param):
#         pr = np.zeros(param.shape, dtype=param.dtype)
#         pr[:ndim ** 2] = param[:ndim ** 2][::-1]
#         pr[ndim ** 2:] = param[ndim ** 2:][::-1]
#         param = pr
#
#         pr = np.reshape(param[: ndim ** 2], (ndim, ndim), order='C')
#         pr = np.concatenate([pr, np.array([param[ndim ** 2:]]).swapaxes(0, 1)], axis=1)
#         return pr
#
#     def _c2m(param):
#         return np.reshape(param, (ndim, ndim + 1), order='C')
#
#     def _f2m(param):
#         return np.reshape(param, (ndim, ndim + 1), order='F')
#
#     def _m2c(param):
#         return param.flatten(order='C')
#
#     def _m2f(param):
#         return param.flatten(order='F')
#
#     def _change_order(params):
#         if param_order == 'elastix':
#             params = _elastix2m(params)
#         if param_order == 'C':
#             params = _c2m(params)
#         if param_order == 'F':
#             params = _f2m(params)
#         if save_order == 'C':
#             params = _m2c(params)
#         if save_order == 'F':
#             params = _m2f(params)
#         return params
#
#     if verbose:
#         print(f'parameters.shape = {parameters.shape}')
#         print(f'parameters.ndim = {parameters.ndim}')
#
#     if param_order != save_order:
#         if (param_order != 'M' and parameters.ndim == 2) or (param_order == 'M' and parameters.ndim == 3):
#             parameters = parameters.tolist()
#             for idx, p in enumerate(parameters):
#                 parameters[idx] = _change_order(np.array(p))
#         else:
#             parameters = _change_order(parameters)
#
#     import json
#
#     if verbose:
#         print(f'parameters.shape = {parameters.shape}')
#
#     if out_filepath is not None:
#         with open(out_filepath, mode='w') as f:
#             json.dump(parameters.tolist(), f, indent=2)
#
#     return parameters


def get_affine_rotation_parameters(euler_angles):
    return sitk.Euler3DTransform((0, 0, 0), *euler_angles).GetMatrix()


def make_auto_mask(image, disk_size=6, method='non-zero', variance_filter_size=3, variance_thresh=10):

    mask = None
    if method == 'non-zero':
        mask = (image > 0).astype('uint8')
    if method == 'variance':
        if image.ndim == 3:
            raise NotImplementedError('Variance-based auto-masking only implemented for 2D')
        from squirrel.library.scaling import scale_image_nearest
        from scipy.ndimage import generic_filter
        mask_t = (generic_filter(scale_image_nearest(image, scale_factors=[0.25, 0.25]), np.var, size=variance_filter_size) > variance_thresh).astype('uint8')
        mask_t = scale_image_nearest(mask_t, scale_factors=[4, 4])
        mask = np.zeros(image.shape, dtype='uint8')
        mask[:mask_t.shape[0], :mask_t.shape[1]] = mask_t
        mask[image == 0] = 0
    if mask is None:
        raise ValueError(f'Invalid auto-mask method: {method}')

    if image.ndim == 2:
        from skimage.morphology import binary_closing, disk
        footprint = disk(disk_size)
        mask = binary_closing(mask, footprint).astype('uint8')
    elif image.ndim == 3:
        from skimage.morphology import binary_closing, ball
        footprint = ball(disk_size)
        mask = binary_closing(mask, footprint).astype('uint8')
    else:
        raise RuntimeError('Only 2D and 3D images allowed!')

    return mask


def compute_mattes_mi(fixed_image, moving_image, bins=50, mask=None, sampling_fraction=0.2):
    """
    Compute the Mattes Mutual Information score between two images, optionally restricted to a mask.

    Parameters
    ----------
    fixed_image : sitk.Image
        Fixed/reference image (SimpleITK image, float32 recommended).
    moving_image : sitk.Image
        Moving image (SimpleITK image, float32 recommended).
    bins : int, optional
        Number of histogram bins (default=50).
    mask : sitk.Image, optional
        Binary mask image (same size as fixed). Only voxels where mask != 0 are used.

    Returns
    -------
    float
        Mattes Mutual Information score (ITK-style, typically negative).
    """
    import SimpleITK as sitk

    if type(fixed_image) == np.ndarray:
        fixed_image = sitk.GetImageFromArray(fixed_image.astype('float32'))
    if type(moving_image) == np.ndarray:
        moving_image = sitk.GetImageFromArray(moving_image.astype('float32'))
    if mask is not None and type(mask) == np.ndarray:
        mask = sitk.GetImageFromArray(mask)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)

    if mask is not None:
        registration.SetMetricFixedMask(mask)

    # Add voxel sampling to avoid scaling with image/mask size
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(sampling_fraction)
    # registration.SetMetricSamplingSeed(42)  # reproducibility

    identity = sitk.Transform(fixed_image.GetDimension(), sitk.sitkIdentity)
    registration.SetInitialTransform(identity, inPlace=False)

    score = registration.MetricEvaluate(fixed_image, moving_image)
    return score


def initialize_offsets_with_xcorr(
        moving_img,
        fixed_img,
        binning=16,
        gaussian_sigma=1.0,
        mi_thresh=-0.6,
        debug_dir=None,
        verbose=False
):
    if type(moving_img) == str:
        from tifffile import imread
        moving_img = imread(moving_img)
    if type(fixed_img) == str:
        from tifffile import imread
        fixed_img = imread(fixed_img)

    ndim = moving_img.ndim

    # moving_img_orig = moving_img.copy()

    from squirrel.library.scaling import scale_image_nearest
    moving_img = scale_image_nearest(moving_img, [1/binning] * ndim)
    fixed_img = scale_image_nearest(fixed_img, [1/binning] * ndim)

    from vigra.filters import gaussianSmoothing
    moving_img = gaussianSmoothing(moving_img.astype('float32'), gaussian_sigma).astype('uint16')
    fixed_img = gaussianSmoothing(fixed_img.astype('float32'), gaussian_sigma).astype('uint16')

    fixed_mask = make_auto_mask(fixed_img, disk_size=6, method='non-zero')
    moving_mask = make_auto_mask(moving_img, disk_size=6, method='non-zero')

    from skimage.registration import phase_cross_correlation
    offsets = phase_cross_correlation(
        fixed_img, moving_img,
        reference_mask=fixed_mask,
        moving_mask=moving_mask,
        upsample_factor=1,
        normalization=None
    )[0]

    from scipy.ndimage.interpolation import shift
    result_image = shift(moving_img, offsets)

    # if verbose:
    #     print(f'offsets = {offsets}')

    if debug_dir is not None:
        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)
        from matplotlib import pyplot as plt
        if ndim == 2:
            plt.imshow((np.array((fixed_img, result_image, result_image)).astype('float32') / fixed_img.max() * 255).astype('uint8').transpose((1, 2, 0)))
            plt.savefig(os.path.join(debug_dir, 'registered_img_overlay.png'))
            plt.close()
        if ndim == 3:
            plt.imshow(
                (
                    np.array((fixed_img, result_image, result_image)).astype('float32') / fixed_img.max() * 255
                ).astype('uint8')[:, int(fixed_img.shape[0] / 2)].transpose((1, 2, 0))
            )
            plt.savefig(os.path.join(debug_dir, 'registered_img_overlay.png'))
            plt.close()

    mi_score = compute_mattes_mi(fixed_img, result_image)  #, mask=this_mask)
    print(f'Found initial offset: {offsets} -- MI score: {mi_score}')

    if mi_score > mi_thresh:
        print(f'Warning: initialization score is too high!\n'
              f'   Ignoring the computed initialization offsets and trying without ...')
        offsets = [0., 0.]

    offsets_unbinned = np.array(offsets) * binning

    from squirrel.library.affine_matrices import AffineMatrix
    return AffineMatrix(translation=-offsets_unbinned)


def initialize_offsets(
        moving_img,
        fixed_img,
        binning=32,
        spacing=256,
        elx_binning=4,
        elx_max_iters=32,
        mi_thresh=-0.8,
        gaussian_sigma=1.0,
        debug_dir=None,
        n_workers=1,
        verbose=False
):
    def _compute_overlapping_offsets(bin_mask_fixed, bin_mask_moving, spacing=64, iou_thresh=0.25):
        """
        Find all integer offsets where two binary masks overlap,
        given a grid spacing (in pixels/voxels).

        Works for both 2D and 3D masks.

        Parameters
        ----------
        bin_mask_fixed : np.ndarray
            Binary mask (2D or 3D) of the fixed image.
        bin_mask_moving : np.ndarray
            Binary mask (2D or 3D) of the moving image (same shape as fixed).
        spacing : int
            Step size in pixels/voxels for the grid.
        iou_thresh : float
            Minimum intersection-over-union (IoU) to keep an offset.

        Returns
        -------
        list of tuple
            List of offsets (dx, dy, [dz]) depending on dimension.
        list of float
            Corresponding Euclidean distances.
        """
        from squirrel.library.scores import intersection_over_union

        assert bin_mask_fixed.shape == bin_mask_moving.shape, "Masks must have same shape"
        shape = bin_mask_fixed.shape
        ndim = bin_mask_fixed.ndim

        # Build ranges of offsets for each dimension
        offset_ranges = [
            range(-s + spacing, s, spacing)
            for s in shape
        ]

        offsets = []
        for delta in np.ndindex(*[len(r) for r in offset_ranges]):
            shift = tuple(offset_ranges[d][i] for d, i in enumerate(delta))

            # compute slices for fixed and moving masks
            slices_fixed = []
            slices_moving = []
            valid = True
            for d, sh in enumerate(shift):
                dim_len = shape[d]

                f0, f1 = max(0, sh), min(dim_len, dim_len + sh)
                m0, m1 = max(0, -sh), min(dim_len, dim_len - sh)

                if f0 >= f1 or m0 >= m1:  # no overlap
                    valid = False
                    break

                slices_fixed.append(slice(f0, f1))
                slices_moving.append(slice(m0, m1))

            if not valid:
                continue

            sub_fixed = bin_mask_fixed[tuple(slices_fixed)]
            sub_moving = bin_mask_moving[tuple(slices_moving)]

            iou = intersection_over_union(sub_fixed, sub_moving)

            if iou > iou_thresh:
                offsets.append(shift)

        # Ensure (0,...,0) is always included
        zero_offset = tuple(0 for _ in range(ndim))
        if zero_offset not in offsets:
            offsets.insert(0, zero_offset)

        # Compute distances
        distances = [np.linalg.norm(off) for off in offsets]

        # Sort by distance
        sorted_pairs = sorted(zip(offsets, distances), key=lambda p: p[1])
        offsets, distances = zip(*sorted_pairs)

        return list(offsets), list(distances)

    if type(moving_img) == str:
        from tifffile import imread
        moving_img = imread(moving_img)
    if type(fixed_img) == str:
        from tifffile import imread
        fixed_img = imread(fixed_img)

    ndim = moving_img.ndim

    assert spacing % binning == 0
    spacing = int(spacing / binning)

    import SimpleITK as sitk
    parameter_map = sitk.GetDefaultParameterMap('translation')
    parameter_map['ImagePyramidSchedule'] = [str(elx_binning)] * ndim
    parameter_map['NumberOfResolutions'] = ["1.000000"]
    parameter_map['ErodeMask'] = ['true']
    parameter_map['MaximumNumberOfIterations'] = [str(elx_max_iters)]

    from squirrel.library.scaling import scale_image_nearest
    moving_img = scale_image_nearest(moving_img, [1/binning] * ndim)
    fixed_img = scale_image_nearest(fixed_img, [1/binning] * ndim)

    from vigra.filters import gaussianSmoothing
    moving_img = gaussianSmoothing(moving_img.astype('float32'), gaussian_sigma).astype('uint16')
    fixed_img = gaussianSmoothing(fixed_img.astype('float32'), gaussian_sigma).astype('uint16')

    fixed_mask = make_auto_mask(fixed_img, disk_size=6, method='non-zero')
    moving_mask = make_auto_mask(moving_img, disk_size=6, method='non-zero')

    offsets, distances = _compute_overlapping_offsets(fixed_mask, moving_mask, spacing=spacing)

    from squirrel.library.affine_matrices import AffineMatrix

    max_offsets = np.max(offsets, axis=0)
    min_offsets = np.min(offsets, axis=0)
    print(max_offsets, min_offsets)
    result_matrix = np.zeros(((max_offsets - min_offsets) / spacing).astype(int) + 1)

    best_score = 0
    best_registered_img = None
    best_offset = [0, 0]

    best_transform_params = None

    for idx, offset in enumerate(offsets):

        this_offset = AffineMatrix(translation=offset)
        this_moving_img = apply_transforms_on_image(moving_img, [this_offset])
        this_moving_mask = apply_transforms_on_image(moving_mask, [this_offset]).astype('uint8')
        this_mask = this_moving_mask * fixed_mask

        try:
            this_transform_params, _ = register(
                fixed_img,
                this_moving_img,
                parameter_map,
                None,
                mask=this_mask,
                # moving_mask=this_moving_mask,
                # fixed_mask=fixed_mask,
                n_workers=n_workers,
                return_result_image=False,
                verbose=False
            )
        except RuntimeError:
            continue
        registered_img = apply_transforms_on_image(moving_img, [this_offset, this_transform_params])

        mi_fixed_vs_registered = compute_mattes_mi(fixed_img, registered_img, mask=this_mask)

        if mi_fixed_vs_registered < best_score:
            best_score = mi_fixed_vs_registered
            if debug_dir is not None:
                best_registered_img = registered_img.copy()
            best_offset = offset
            best_transform_params = this_offset * AffineMatrix(elastix_parameters=this_transform_params)

        result_matrix[tuple(((offset - min_offsets) / spacing).astype(int))] = mi_fixed_vs_registered
        if mi_fixed_vs_registered < mi_thresh:
            break

    if idx == len(offsets) - 1:
        print(f'Break criterion of score < {mi_thresh} never reached, using position with minimal score')

    if best_transform_params is None:
        from tifffile import imwrite
        import random
        crash_dir = os.path.join(os.getcwd(), f'crash_{random.randint(0, 9999)}')
        if not os.path.exists(crash_dir):
            os.mkdir(crash_dir)
        imwrite(os.path.join(crash_dir, 'moving.tif'), moving_img)
        imwrite(os.path.join(crash_dir, 'fixed.tif'), fixed_img)
        print(f'Written crashing data to {crash_dir}')
        raise RuntimeError('Initialization failed!')

    best_offset_unbinned = np.array(best_offset) * binning
    best_transform_params_unbinned = best_transform_params.get_scaled(binning)

    if verbose:
        print(f'best_score = {best_score}')
        print(f'best_offset = {best_offset}')
        print(f'best_offset_unbinned = {best_offset_unbinned}')
        print(f'best_transform_params = {best_transform_params.get_matrix()}')
        print(f'best_transform_params_unbinned = {best_transform_params_unbinned.get_matrix()}')

    if debug_dir is not None:
        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)
        from matplotlib import pyplot as plt
        if ndim == 2:
            plt.imshow((np.array((fixed_img, best_registered_img, best_registered_img)).astype('float32') / fixed_img.max() * 255).astype('uint8').transpose((1, 2, 0)))
            plt.savefig(os.path.join(debug_dir, 'registered_img_overlay.png'))
            plt.close()
            plt.imshow(result_matrix)
            plt.savefig(os.path.join(debug_dir, 'mi_score_matrix.png'))
        if ndim == 3:
            plt.imshow(
                (
                    np.array((fixed_img, best_registered_img, best_registered_img)).astype('float32') / fixed_img.max() * 255
                ).astype('uint8')[:, int(fixed_img.shape[0] / 2)].transpose((1, 2, 0))
            )
            plt.savefig(os.path.join(debug_dir, 'registered_img_overlay.png'))
            plt.close()
            plt.imshow(result_matrix[int(result_matrix.shape[0] / 2)])
            plt.savefig(os.path.join(debug_dir, 'mi_score_matrix.png'))
        plt.close()

    print(f'Found initial offset: {best_transform_params_unbinned.get_translation()} -- MI score: {best_score}')

    return best_transform_params_unbinned


def register(
        fixed_image,
        moving_image,
        parameter_map,
        out_dir,
        mask=None,
        moving_mask=None,
        fixed_mask=None,
        n_workers=1,
        return_result_image=False,
        verbose=False
):
    import SimpleITK as sitk

    if type(fixed_image) == np.ndarray:
        if verbose:
            print(f'Getting fixed image from array with shape = {fixed_image.shape}')
        fixed_image = sitk.GetImageFromArray(fixed_image)
    if type(moving_image) == np.ndarray:
        if verbose:
            print(f'Getting moving image from array with shape = {moving_image.shape}')
        moving_image = sitk.GetImageFromArray(moving_image)

    if verbose:
        print(f'fixed_image.GetSize() = {fixed_image.GetSize()}')
        print(f'moving_image.GetSize() = {moving_image.GetSize()}')

    # Set the input images
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)
    if mask is not None:
        mask = sitk.GetImageFromArray(mask)
        elastixImageFilter.SetFixedMask(mask)
        elastixImageFilter.SetMovingMask(mask)
        parameter_map['ErodeMask'] = ['true']
    if fixed_mask is not None:
        fixed_mask = sitk.GetImageFromArray(fixed_mask)
        elastixImageFilter.SetFixedMask(fixed_mask)
        assert moving_mask is not None
        moving_mask = sitk.GetImageFromArray(moving_mask)
        elastixImageFilter.SetMovingMask(moving_mask)
        parameter_map['ErodeMask'] = ['true']
    if out_dir is not None:
        elastixImageFilter.SetOutputDirectory(out_dir)
    elastixImageFilter.LogToConsoleOff()
    if verbose:
        elastixImageFilter.LogToConsoleOn()

    elastixImageFilter.SetParameterMap(parameter_map)

    if verbose:
        print(f'Running Elastix with these parameters:')
        elastixImageFilter.PrintParameterMap()
    elastixImageFilter.SetNumberOfThreads(n_workers)
    elastixImageFilter.Execute()
    result_image = None
    if return_result_image:
        result_image = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())

    elastix_transform_param_map = elastixImageFilter.GetTransformParameterMap()[0]

    return elastix_transform_param_map, result_image


def register_with_elastix(
        fixed_image, moving_image,
        transform=None,
        automatic_transform_initialization=False,
        out_dir=None,
        params_to_origin=False,
        auto_mask='non-zero',  # This really should be used!
        number_of_spatial_samples=None,
        maximum_number_of_iterations=None,
        number_of_resolutions=None,
        return_result_image=False,
        initialize_offsets_method=None,
        initialize_offsets_kwargs=None,
        parameter_map=None,
        median_radius=0,
        gaussian_sigma=0.,
        use_edges=False,
        use_clahe=False,
        crop_to_bounds_off=False,
        n_workers=os.cpu_count(),
        normalize_images=True,
        result_to_disk='',
        debug_dirpath=None,
        verbose=False
):

    if verbose:
        print('Running register_with_elastix with:')
        print(f'automatic_transform_initialization={automatic_transform_initialization}')
        print(f'out_dir={out_dir}')
        print(f'params_to_origin={params_to_origin}')
        print(f'auto_mask={auto_mask}')
        print(f'number_of_spatial_samples={number_of_spatial_samples}')
        print(f'maximum_number_of_iterations={maximum_number_of_iterations}')
        print(f'number_of_resolutions={number_of_resolutions}')
        print(f'return_result_image={return_result_image}')
        print(f'initialize_offsets_method={initialize_offsets_method}')
        print(f'initialize_offsets_kwargs={initialize_offsets_kwargs}')
        print(f'parameter_map={parameter_map}')
        print(f'median_radius={median_radius}')
        print(f'gaussian_sigma={gaussian_sigma}')
        print(f'use_edges={use_edges}')
        print(f'use_clahe={use_clahe}')
        print(f'crop_to_bounds_off={crop_to_bounds_off}')
        print(f'n_workers={n_workers}')
        print(f'normalize_images={normalize_images}')
        print(f'debug_dirpath = {debug_dirpath}')

    import SimpleITK as sitk

    def _normalize_parameter_map(pmap, transform):
        if pmap is None:
            assert transform is not None, 'Either parameter_map or transform must be specified!'
            # Set the parameters
            pmap = sitk.GetDefaultParameterMap(transform if transform != 'SimilarityTransform' else 'rigid')
            pmap['AutomaticTransformInitialization'] = ['true' if automatic_transform_initialization else 'false']
            if number_of_spatial_samples is not None:
                pmap['NumberOfSpatialSamples'] = (str(number_of_spatial_samples),)
                pmap['NumberOfSamplesForExactGradient'] = (str(number_of_spatial_samples * 2),)
            if maximum_number_of_iterations is not None:
                pmap['MaximumNumberOfIterations'] = (str(maximum_number_of_iterations),)
            if number_of_resolutions is not None:
                pmap['NumberOfResolutions'] = (str(number_of_resolutions),)
            if transform == 'SimilarityTransform':
                pmap['Transform'] = ['SimilarityTransform']
        if type(pmap) == str:
            from SimpleITK import ReadParameterFile
            assert os.path.exists(pmap)
            pmap = ReadParameterFile(pmap)
        if transform is None:
            assert pmap is not None,  'Either parameter_map or transform must be specified!'
            # transform = 'translation' if pmap['Transform'][0] == 'TranslationTransform' else pmap['Transform']
        transform = pmap['Transform'][0]
        return pmap, transform

    def _load_image(img):
        if verbose:
            print(f'type(img) = {type(img)}')
        if type(img) == str:
            from tifffile import imread
            img = imread(img)
        dtype = img.dtype
        assert dtype == 'uint8' or dtype == 'uint16', \
            f'Only allowing 8 or 16 bit unsigned integer images. Fixed image has dtype = {dtype}'
        return img

    def _crop_to_bounds(fixed, moving):
        """
        Crop `fixed` and `moving` arrays to a common bounding box.

        Returns:
            fixed_cropped, moving_cropped, offset
        """
        from squirrel.library.image import get_bounds

        # Get bounding boxes for both arrays
        bounds_fixed = np.array(get_bounds(fixed, return_ints=True))
        bounds_moving = np.array(get_bounds(moving, return_ints=True))

        ndim = fixed.ndim

        # Split bounds into mins and maxs
        mins = np.minimum(bounds_fixed[:ndim], bounds_moving[:ndim]).astype(int)
        maxs = np.maximum(bounds_fixed[ndim:], bounds_moving[ndim:]).astype(int)
        offset = mins

        # Create slicing object for each dimension
        bounds = tuple(slice(lo, hi) for lo, hi in zip(mins, maxs))

        # Crop arrays
        fixed_cropped = fixed[bounds]
        moving_cropped = moving[bounds]

        return fixed_cropped, moving_cropped, offset

    def _generate_mask(img):

        mask = make_auto_mask(img, disk_size=6, method=auto_mask)
        if gaussian_sigma > 0:
            from skimage.morphology import erosion
            import math
            if mask.ndim == 2:
                from skimage.morphology import disk
                mask = erosion(mask, footprint=disk(int(math.ceil(3 * gaussian_sigma))))
            else:
                from skimage.morphology import ball
                mask = erosion(mask, footprint=ball(int(math.ceil(3 * gaussian_sigma))))
        if use_edges:
            img[mask == 0] = 0
        return mask

    def _pre_process_image(img):

        dtype = img.dtype

        if use_clahe:
            from squirrel.library.normalization import clahe_on_image
            img = clahe_on_image(img)
        if median_radius > 0:
            from skimage.filters import median
            from skimage.morphology import disk
            img = median(img, footprint=disk(median_radius))
        if gaussian_sigma > 0:
            from skimage.filters import gaussian
            img = gaussian(img.astype(float), gaussian_sigma).astype(dtype)
        if use_edges:
            from skimage.filters import sobel
            img = sobel(img.astype(float))

        mask = _generate_mask(img) if auto_mask is not None else None

        if normalize_images:
            assert type(fixed_image) == np.ndarray
            from squirrel.library.data import norm_full_range
            img = norm_full_range(img, (0.05, 0.95), ignore_zeros=False, mask=mask, cast_8bit=True)

        return img, mask

    def _initialize_offsets(fixed, moving, kwargs):
        if kwargs is None:
            kwargs = {}
        from squirrel.library.affine_matrices import AffineMatrix
        offsets = AffineMatrix(translation=[0.] * fixed.ndim)
        if initialize_offsets_method is not None and initialize_offsets_method != 'none':
            if verbose:
                print(f'Running initialization to cope with big jumps!')
            assert type(fixed) == np.ndarray
            assert type(moving) == np.ndarray
            if transform == 'bspline':
                raise NotImplementedError('Big jump fixing not implemented for bspline transformation!')
            if initialize_offsets_method == 'init_xcorr':
                offsets = initialize_offsets_with_xcorr(
                    moving, fixed,
                    binning=16 if 'binning' not in kwargs else kwargs['binning'],
                    mi_thresh=-0.6 if 'mi_thresh' not in kwargs else kwargs['mi_thresh'],
                    debug_dir=debug_dirpath
                )
            elif initialize_offsets_method == 'init_elx':
                offsets = initialize_offsets(
                    moving, fixed,
                    binning=32 if 'binning' not in kwargs else kwargs['binning'],
                    spacing=256 if 'spacing' not in kwargs else kwargs['spacing'],
                    elx_binning=4 if 'elx_binning' not in kwargs else kwargs['elx_binning'],
                    elx_max_iters=32 if 'elx_max_iters' not in kwargs else kwargs['elx_max_iters'],
                    mi_thresh=-0.8 if 'mi_thresh' not in kwargs else kwargs['mi_thresh'],
                    gaussian_sigma=1.0 if 'gaussian_sigma' not in kwargs else kwargs['gaussian_sigma'],
                    debug_dir=debug_dirpath
                )
            else:
                raise ValueError(f'Invalid initialization method: {initialize_offsets_method} is not in ["xcorr", "init_elx"]')
            # offsets = np.array(offsets)
        return offsets

    def _finalize_result_transform_parameters(elx_param_map, bounds_offset):
        result_transform_parameters = elx_param_map['TransformParameters']
        try:
            pivot = np.array([float(x) for x in elx_param_map['CenterOfRotationPoint']])[::-1]
        except IndexError:
            pivot = np.array([0.] * fixed_image.ndim)
        pivot += bounds_offset

        from squirrel.library.affine_matrices import AffineMatrix
        result_matrix = AffineMatrix(
            elastix_parameters=[transform, [float(x) for x in result_transform_parameters]]
        )
        result_matrix = result_matrix * pre_fix_offsets
        result_matrix.set_pivot(pivot)

        if params_to_origin:
            if verbose:
                print(f'shifting params to origin')
                print(f'result_matrix.get_pivot() = {result_matrix.get_pivot()}')
            result_matrix.shift_pivot_to_origin()
        return result_matrix

    def _debug_step(fixed, moving, name):
        if debug_dirpath is not None:
            os.makedirs(debug_dirpath, exist_ok=True)
            from tifffile import imwrite, imsave
            imwrite(os.path.join(debug_dirpath, f'{name}-fixed.tif'), fixed)
            imwrite(os.path.join(debug_dirpath, f'{name}-moving.tif'), moving)
            combined = np.array([
                fixed,
                moving,
                moving
            ])
            imwrite(os.path.join(debug_dirpath, f'{name}-combined.tif'), combined)

    from squirrel.library.image import assert_equal_shape

    # The inputs can be string (filepath) or an image. Make sure it's loaded
    print(f'Fetching data ...')
    fixed_image = _load_image(fixed_image)
    moving_image = _load_image(moving_image)
    fixed_image, moving_image = assert_equal_shape(fixed_image, moving_image)
    assert moving_image.dtype == fixed_image.dtype, \
        f'fixed and moving images must have the same data type: {fixed_image.dtype} != {moving_image.dtype}'
    _debug_step(fixed_image, moving_image, '00-input')

    # import SimpleITK as sitk

    # Make sure the parameter map is an Elastix parameter map instance
    print(f'Checking Elastix parameter map ...')
    parameter_map, transform = _normalize_parameter_map(parameter_map, transform)
    if debug_dirpath is not None:
        from SimpleITK import WriteParameterFile
        WriteParameterFile(parameter_map, os.path.join(debug_dirpath, 'elastix_parameters.txt'))
    assert transform == parameter_map['Transform'][0]

    # Crop the input images to their joint bounding box (saves memory and speeds up processing below)
    print('Cropping data ...')
    bounds_offset = np.array([0.] * fixed_image.ndim)
    crop_to_bounds = auto_mask is not None
    if crop_to_bounds_off or transform in ['bspline', 'BSplineTransform']:
        crop_to_bounds = False
    if crop_to_bounds:
        fixed_image, moving_image, bounds_offset = _crop_to_bounds(fixed_image, moving_image)
    _debug_step(fixed_image, moving_image, '01-cropped')

    # Pre-process the images
    print(f'Pre-processing images ...')
    fixed_image, fixed_mask = _pre_process_image(fixed_image)
    moving_image, moving_mask = _pre_process_image(moving_image)
    _debug_step(fixed_image, moving_image, '02-pre-processed')
    _debug_step(fixed_mask, moving_mask, '03-mask')

    # Initialize offsets in case of a larger initial shift
    print(f'Initializing offsets ...')
    pre_fix_offsets = _initialize_offsets(fixed_image, moving_image, initialize_offsets_kwargs)

    # Shift the moving image and it's mask
    print('Applying initialization ...')
    if not all(pre_fix_offsets.get_translation() == 0):
        moving_image = apply_transforms_on_image(moving_image, [pre_fix_offsets])
        if moving_mask is not None:
            moving_mask = apply_transforms_on_image(moving_mask, [pre_fix_offsets])
    _debug_step(fixed_image, moving_image, '04-after-init')

    # Run registration
    print(f'Running registration ...')
    mask = (moving_mask * fixed_mask).astype('uint8') if moving_mask is not None else None
    elastix_transform_param_map, result_image = register(
        fixed_image,
        moving_image,
        parameter_map,
        out_dir,
        mask=mask,
        # fixed_mask=fixed_mask,
        # moving_mask=moving_mask,
        n_workers=n_workers,
        return_result_image=return_result_image,
        verbose=verbose
    )

    # Return an affine matrix object for any rigid or affine transformation
    print(f'Finalizing output transformation ...')
    if transform not in ['bspline', 'BSplineTransform']:
        result_matrix = _finalize_result_transform_parameters(elastix_transform_param_map, bounds_offset)

        if result_to_disk:
            result_matrix.to_file(result_to_disk)
            return result_to_disk
        return result_matrix, result_image

    if result_to_disk:
        sitk.WriteParameterFile(elastix_transform_param_map, result_to_disk)
        return result_to_disk
    # Return the elastix parameters for non-rigid registration
    return elastix_transform_param_map, result_image


def slice_wise_stack_to_stack_alignment(
        moving_stack,
        fixed_stack,
        transform='affine',
        automatic_transform_initialization=False,
        out_dir=None,
        auto_mask='non-zero',
        median_radius=0,
        gaussian_sigma=0.,
        number_of_spatial_samples=None,
        maximum_number_of_iterations=None,
        number_of_resolutions=None,
        return_result_image=False,
        initialize_offsets_method=None,
        initialize_offsets_kwargs=None,
        parameter_map=None,
        crop_to_bounds_off=False,
        normalize_images=False,
        n_workers=os.cpu_count(),
        quiet=False,
        verbose=False
):

    if transform in ['translation', 'affine']:
        from ..library.affine_matrices import AffineStack
        result_transforms = AffineStack(is_sequenced=True, pivot=[0., 0.])
    elif transform in ['bspline', 'BSplineTransform']:
        result_transforms = ElastixStack()
    else:
        raise ValueError(f'Invalid transform: {transform}')
    result_stack = []

    if n_workers == 1:
        for zidx, z_slice_moving in enumerate(moving_stack):
            if not quiet:
                print(f'{zidx} / {len(moving_stack) - 1}')
            z_slice_fixed = fixed_stack[zidx]

            result_matrix, result_image = register_with_elastix(
                z_slice_fixed, z_slice_moving,
                transform=transform,
                automatic_transform_initialization=automatic_transform_initialization,
                out_dir=out_dir,
                auto_mask=auto_mask,
                number_of_spatial_samples=number_of_spatial_samples,
                maximum_number_of_iterations=maximum_number_of_iterations,
                number_of_resolutions=number_of_resolutions,
                return_result_image=return_result_image,
                initialize_offsets_method=initialize_offsets_method,
                initialize_offsets_kwargs=initialize_offsets_kwargs,
                parameter_map=parameter_map,
                median_radius=median_radius,
                gaussian_sigma=gaussian_sigma,
                crop_to_bounds_off=crop_to_bounds_off,
                normalize_images=normalize_images,
                verbose=verbose
            )
            if transform not in ['bspline', 'BSplineTransform']:
                result_matrix.shift_pivot_to_origin()

            if result_image is not None:
                result_stack.append(result_image)
            result_transforms.append(result_matrix)

    else:

        import SimpleITK as sitk
        import tempfile

        # parameter_map_filepath = f'./tmp-elx-parameters-{os.getpid()}.txt'
        # parameter_map_filepath = f'./tmp-elx-parameters-{uuid.uuid4().hex}'
        fd, parameter_map_filepath = tempfile.mkstemp(
            prefix=f"tmp-elx-parameters-", suffix='.txt', dir='./'
        )
        os.close(fd)
        if parameter_map is not None:
            sitk.WriteParameterFile(parameter_map, parameter_map_filepath)

        from multiprocessing import Pool
        with Pool(processes=n_workers) as p:
            tasks = []
            for zidx, z_slice_moving in enumerate(moving_stack):
                # if not quiet:
                #     print(f'{zidx} / {len(moving_stack) - 1}')
                # results_filepath = f'./tmp-elx-result-{zidx}-{os.getpid()}.txt'
                # results_filepath = f'./tmp-elx-result-{zidx}-{uuid.uuid4().hex}.txt'
                fd, results_filepath = tempfile.mkstemp(
                    prefix=f"tmp-elx-result-{zidx}-", suffix='.txt', dir='./'
                )
                os.close(fd)
                z_slice_fixed = fixed_stack[zidx]
                tasks.append(p.apply_async(
                    register_with_elastix, (
                        z_slice_fixed, z_slice_moving
                    ),
                    dict(
                        transform=transform,
                        automatic_transform_initialization=automatic_transform_initialization,
                        out_dir=out_dir,
                        auto_mask=auto_mask,
                        number_of_spatial_samples=number_of_spatial_samples,
                        maximum_number_of_iterations=maximum_number_of_iterations,
                        number_of_resolutions=number_of_resolutions,
                        return_result_image=return_result_image,
                        initialize_offsets_method=initialize_offsets_method,
                        initialize_offsets_kwargs=initialize_offsets_kwargs,
                        parameter_map=parameter_map_filepath,
                        median_radius=median_radius,
                        gaussian_sigma=gaussian_sigma,
                        crop_to_bounds_off=crop_to_bounds_off,
                        normalize_images=normalize_images,
                        result_to_disk=results_filepath if type(result_transforms) == ElastixStack else '',
                        verbose=verbose
                    )
                ))

            results = []
            for tidx, task in enumerate(tasks):
                results.append(task.get())
                print(f'{tidx} / {len(tasks) - 1}')
            # results = [task.get() for task in tasks]

        if type(result_transforms) == ElastixStack:
            for result_filepath in results:
                result_matrix = sitk.ReadParameterFile(result_filepath)
                result_transforms.append(result_matrix)
                os.remove(result_filepath)
        else:
            for result_matrix, result_image in results:
                result_matrix.shift_pivot_to_origin()
                if result_image is not None:
                    result_stack.append(result_image)
                result_transforms.append(result_matrix)
        os.remove(parameter_map_filepath)

    return result_transforms, result_stack


def translation_to_c(parameters):
    from ..library.transformation import setup_translation_matrix
    ndim = 0
    if len(parameters) == 2:
        ndim = 2
    if len(parameters) == 3:
        ndim = 3
    return setup_translation_matrix([float(x) for x in parameters[::-1]], ndim=ndim).flatten()


def affine_to_c(parameters):

    parameters = np.array(parameters)
    ndim = 0
    if len(parameters) == 6:
        ndim = 2
    if len(parameters) == 12:
        ndim = 3
    assert ndim in [2, 3], f'Invalid parameters: {parameters}'

    pr = np.zeros(parameters.shape, dtype=parameters.dtype)
    pr[:ndim ** 2] = parameters[:ndim ** 2][::-1]
    pr[ndim ** 2:] = parameters[ndim ** 2:][::-1]
    param = pr

    pr = np.reshape(param[: ndim ** 2], (ndim, ndim), order='C')
    pr = np.concatenate([pr, np.array([param[ndim ** 2:]]).swapaxes(0, 1)], axis=1)
    return pr.flatten()


def rigid_to_c(parameters):
    raise NotImplementedError


def similarity_to_c(parameters):
    raise NotImplementedError


def elastix_to_c(transform, parameters):
    func = None
    if transform in ['translation', 'TranslationTransform']:
        func = translation_to_c
    if transform == 'affine':
        func = affine_to_c
    if transform == 'rigid':
        func = rigid_to_c
    if transform == 'SimilarityTransform':
        func = similarity_to_c
    if func is None:
        raise ValueError(f'Invalid transform: {transform}')
    return func(parameters)


def c_to_elastix(parameters):
    parameters = np.array(parameters)

    ndim = 0
    if len(parameters) == 6:
        ndim = 2
    if len(parameters) == 12:
        ndim = 3
    assert ndim in [2, 3], f'Invalid parameters: {parameters}'

    out_parameters = np.zeros(parameters.shape, dtype=parameters.dtype)

    pr = np.array([parameters[((ndim + 1) * idx): ((ndim + 1) * (idx + 1))] for idx in range(ndim)])
    out_parameters[: ndim ** 2] = pr[:, :ndim].flatten()[::-1]
    out_parameters[ndim ** 2:] = pr[:, ndim][::-1]

    return out_parameters


class ElastixStack:

    def __init__(
            self,
            stack=None,
            dirpath=None,
            pattern='*.txt',
            image_shape=None
    ):
        self._dirpath = None
        self._pattern = None
        self._stack = None
        self._it = 0
        if stack is not None:
            self.set_from_stack(stack, image_shape=image_shape)
        if dirpath is not None:
            self.set_from_dir(dirpath, pattern)

    def set_from_stack(self, stack, image_shape=None):
        # assert type(stack) is list or type(stack) is tuple, 'Only accepting lists or tuples'
        from SimpleITK import ParameterMap
        if isinstance(stack[0], ParameterMap):
            self._stack = list(stack)
            return
        from squirrel.library.affine_matrices import AffineStack
        if isinstance(stack, AffineStack):
            self._stack = [
                x.to_elastix_affine(return_parameter_map=True, shape=image_shape) for x in stack
            ]

    def set_from_dir(self, dirpath, pattern):
        from SimpleITK import ReadParameterFile
        import glob
        filepaths = sorted(glob.glob(os.path.join(dirpath, pattern)))
        self._stack = [ReadParameterFile(filepath) for filepath in filepaths]
        self._dirpath = dirpath
        self._pattern = pattern

    def to_file(self, dirpath):
        from SimpleITK import WriteParameterFile
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        for idx, transform in enumerate(self._stack):
            filepath = os.path.join(dirpath, 'transform_{:05d}.txt'.format(idx))
            WriteParameterFile(transform, filepath)

    def __getitem__(self, item):
        if self._stack is not None:
            return self._stack[item]
        return []

    def __setitem__(self, key, value):
        self._stack[key] = value

    def __len__(self):
        if self._stack is None:
            return 0
        return len(self._stack)

    def __iter__(self):
        self._it = 0
        return self

    def __next__(self):
        if self._it < len(self):
            x = self[self._it]
            self._it += 1
            return x
        else:
            raise StopIteration

    def append(self, other):
        if isinstance(other, ElastixStack):
            for x in other:
                self.append(x)
            return
        from SimpleITK import ParameterMap
        if isinstance(other, ParameterMap):
            self.set_from_stack(self[:] + [other])
            return
        raise ValueError(f'Invalid type of other: {type(other)}')

    def image_shape(self):
        shapes = []
        for transform in self:
            shapes.append(np.array([int(float(x)) for x in transform['Size']])[::-1])
        shapes = np.array(shapes)
        assert np.all(shapes == shapes[0], axis=0).all(), 'Not all shapes of the transforms are equal - but they should be!'

        return shapes[0]


def load_transform_stack_from_multiple_files(paths, sequence_stack=False):

    stack = ElastixStack(dirpath=paths[0])
    for path in paths[1:]:
        stack.append(ElastixStack(dirpath=path))
    return stack


class ElastixMultiStepStack:

    def __init__(
            self,
            stacks=None,
            image_shape=None
    ):
        self._stacks = None
        self._num_steps = 0
        self._it = 0
        if stacks is not None:
            [self.add_stack(stack, image_shape=image_shape) for stack in stacks]

    def add_stack(self, stack, image_shape=None):
        stack = ElastixStack(stack=stack, image_shape=image_shape)
        if self._stacks is not None:
            assert len(stack) == len(self), 'The length of the added stack does not match'
            self._stacks = [self[idx] + [s] for idx, s in enumerate(stack)]
            self._num_steps += len(stack)
            return
        self._stacks = [[s] for s in stack]
        self._num_steps = len(stack)

    def __len__(self):
        return len(self[:])

    def __getitem__(self, item):
        if self._stacks is None:
            return []
        return self._stacks[item]

    def __setitem__(self, key, value):
        assert len(value) == self._num_steps, 'Number of elements in value does not match number of steps!'
        self._stacks[key] = value

    def __iter__(self):
        self._it = 0
        return self

    def __next__(self):
        if self._it < len(self):
            x = self[self._it]
            self._it += 1
            return x
        else:
            raise StopIteration

    def apply_on_image_stack(
            self,
            image_stack_h,
            target_image_shape=None,
            z_range=None,
            key=None,       # Only needed for multiprocessing
            pattern=None,   # Only needed for multiprocessing
            n_workers=1,
            quiet=False,
            verbose=False
    ):
        from squirrel.library.data import norm_z_range
        from squirrel.library.io import load_data_handle
        if type(image_stack_h) == str:
            h, shape = load_data_handle(image_stack_h, key, pattern)
            z_range = norm_z_range(z_range, len(h))
            dtype = h.dtype
        else:
            z_range = norm_z_range(z_range, len(image_stack_h))
            dtype = image_stack_h.dtype
        result_volume = []
        max_val = np.iinfo(dtype).max
        assert dtype == 'uint8' or dtype == 'uint16'

        if verbose:
            print(f'target_image_shape = {target_image_shape}')

        if n_workers == 1:

            for stack_idx, image_idx in enumerate(range(*z_range)):

                result_volume.append(apply_transforms_on_image_stack_slice(
                    image_stack_h,
                    image_idx,
                    self[stack_idx],
                    target_image_shape=target_image_shape,
                    n_slices=z_range[1],
                    key=key,
                    pattern=pattern,
                    n_workers=1,
                    quiet=quiet,
                    verbose=verbose
                ))

        else:

            import tempfile
            cache_dir = tempfile.mkdtemp()
            transform_fps = self.to_disk(cache_dir)
            from multiprocessing import Pool
            with Pool(processes=n_workers) as p:
                tasks = [
                    p.apply_async(
                        apply_transforms_on_image_stack_slice, (
                            image_stack_h, image_idx, transform_fps[stack_idx]
                        ), dict(
                            target_image_shape=target_image_shape,
                            n_slices=z_range[1],
                            key=key,
                            pattern=pattern,
                            n_workers=1,
                            quiet=quiet,
                            verbose=verbose
                        )
                    )
                    for stack_idx, image_idx in enumerate(range(*z_range))
                ]

                result_volume = [task.get() for task in tasks]

            from shutil import rmtree
            rmtree(cache_dir)

        if verbose:
            print(f'result_volume[0].shape = {result_volume[0].shape}')

        # else:
        #
        #     from concurrent.futures import ThreadPoolExecutor
        #     with ThreadPoolExecutor(max_workers=n_workers) as tpe:
        #         tasks = [
        #             tpe.submit(
        #                 apply_transforms_on_image_stack_slice,
        #                 image_stack_h,
        #                 image_idx,
        #                 self[stack_idx],
        #                 target_image_shape,
        #                 z_range[1],
        #                 1,
        #                 quiet,
        #                 verbose
        #             )
        #             for stack_idx, image_idx in enumerate(range(*z_range))
        #         ]
        #         result_volume = [task.result() for task in tasks]

        return np.clip(np.array(result_volume), 0, max_val).astype(dtype)

    def to_disk(self, dirpath):

        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

        filepaths = []
        import SimpleITK as sitk
        for idx, item in enumerate(self):
            this_filepaths = []
            for jdx, transform in enumerate(item):
                this_filepaths.append(os.path.join(dirpath, '{:05d}_{:02d}.txt'.format(idx, jdx)))
                sitk.WriteParameterFile(transform, this_filepaths[-1])
            filepaths.append(this_filepaths)
        return filepaths


if __name__ == '__main__':

    # a = [1, 2, 3, 4, 5, 6]
    # print(f'a = {a}')
    # b = affine_to_c(a)
    # print(f'b = {b}')
    # c = c_to_elastix(b)
    # print(f'c = {c}')
    #
    # a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # print(f'a = {a}')
    # b = affine_to_c(a)
    # print(f'b = {b}')
    # c = c_to_elastix(b)
    # print(f'c = {c}')

    # es = ElastixStack(dirpath='/media/julian/Data/projects/hennies/amst_devel/240624_snamst_kors_dT/amst.meta/amst/')
    # print(es.image_shape())

    # initialize_offsets(
    #     '/media/julian/Data/projects/hennies/amst_devel/amst2-test-auto-init/tiffs/slice_01926_z=19.3991um.tif',
    #     '/media/julian/Data/projects/hennies/amst_devel/amst2-test-auto-init/tiffs/slice_01927_z=19.4090um.tif',
    #     debug_dir='/media/julian/Data/tmp/init_elx_align/'
    # )

    from tifffile import imread, imwrite
    # moving_img = imread('/media/julian/Data/projects/hennies/amst_devel/amst2-test-auto-init/tiffs/slice_01927_z=19.4090um.tif')
    # fixed_img = imread('/media/julian/Data/projects/hennies/amst_devel/amst2-test-auto-init/tiffs/slice_01926_z=19.3991um.tif')
    # moving_img = imread('/media/julian/Data/projects/hennies/amst_devel/amst2-hela-join-parts/tiffs/slice_01786.tif')
    # fixed_img = imread('/media/julian/Data/projects/hennies/amst_devel/amst2-hela-join-parts/tiffs/slice_01787.tif')

    fixed_img = imread('/media/julian/Data/courses/2025_embo_volume_sem/hela_sl200_455_uint8_cropped_to_data_g4_binz4xy8/slice_00449.tif')
    moving_img = imread('/media/julian/Data/courses/2025_embo_volume_sem/hela_sl200_455_uint8_cropped_to_data_g4_binz4xy8/slice_00450.tif')

    register_with_elastix(
        fixed_img,
        moving_img,
        transform='translation',
        initialize_offsets_method='init_elx',
        initialize_offsets_kwargs=dict(binning=4, spacing=64),
        debug_dirpath='/media/julian/Data/courses/2025_embo_volume_sem/tmp/debug'
    )
