
import numpy as np
import inspect


def fft_highpass(
        image: np.ndarray,
        sigma: (float, float) = (10, 10),
        keep_zeros: bool = False,
        cast_dtype: str = None
) -> np.ndarray:
    """
    Apply an anisotropic high-pass filter to a 2D image using FFT,
    matching a Gaussian high-pass filter with given spatial sigmas.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image
    sigma : float
        Gaussian sigmas along x- and y-axis (pixels)
    keep_zeros : bool
        Pixels that are zero in the input will be zero in the output
    cast_dtype :
        Output will be normalized and casted to the requested dtype (accepted values: "uint8", "uint16")

    Returns
    -------
    result : 2D np.ndarray
        High-pass filtered image
    """

    if cast_dtype not in ['uint8', 'uint16', None]:
        raise ValueError(f'Invalid dtype for casting: {cast_dtype}. Possible values: ["uint8", "uint16", None]')

    mask = None
    if keep_zeros:
        mask = image == 0

    sigma_y, sigma_x = sigma
    ny, nx = image.shape

    # FFT of the image
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)  # center zero frequency

    # Frequency grids (cycles per pixel, shifted to [-0.5, 0.5))
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky)

    # Convert spatial sigma to frequency domain parameter
    sx_freq = nx / (2 * np.pi * sigma_x)
    sy_freq = ny / (2 * np.pi * sigma_y)

    # Anisotropic high-pass filter
    H = 1 - np.exp(-((KX * nx)**2 / sx_freq**2 + (KY * ny)**2 / sy_freq**2))

    # Apply filter
    F_filtered = F_shifted * H

    # Transform back to spatial domain
    result = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))

    if cast_dtype in ['uint8', 'uint16']:
        result -= result.min()
        result = result / result.max() * 255
        result = result.astype('uint8')

    if keep_zeros:
        result[mask] = 0

    return result


class ImageFilter:

    def __init__(self, in_array):
        self._in_array = in_array

    def _check_filter_names(self, filter_names):
        available_filters = self.get_available_filter_names()
        for filter_name in filter_names:
            if filter_name not in available_filters:
                raise ValueError(f'{filter_name} not in available filters: {available_filters}')

    def get_filtered(
            self, filters: list[list[object]], in_array=None
    ) -> np.ndarray:

        filter_names = [filter[0] for filter in filters]
        self._check_filter_names(filter_names)

        result_array = self._in_array.copy() if in_array is None else in_array
        for filter_name, filter_kwargs in filters:
            result_array = getattr(self, filter_name)(result_array, **filter_kwargs)

        return result_array

    def get_filtered_stack(
            self, filters: list[list[object]], n_workers: int = 1
    ) -> np.ndarray:
        if not (2 < self._in_array.ndim <= 4):
            raise RuntimeError(f'in_array.ndim = {self._in_array.ndim}. \n'
                               'in_array needs to have ndim=3 (stack of 2D images) or ndim=4 (stack of 3D images) '
                               'for stack processing.')

        if n_workers == 1:
            result_array = np.array([
                self.get_filtered(filters, in_array=im_slice)
                for im_slice in self._in_array
            ])

        else:
            print(f'Running ImageFilter stack with {n_workers} CPUs')
            from multiprocessing import Pool
            with Pool(processes=n_workers) as p:
                tasks = [
                    p.apply_async(
                        self.get_filtered, (filters,), dict(in_array=im_slice)
                    )
                    for im_slice in self._in_array
                ]
                result_array = np.array([task.get() for task in tasks])

        return result_array

    def get_available_filter_names(self):
        exclude = []
        return [
            name for name, member in inspect.getmembers(self.__class__, predicate=inspect.isfunction)
            if not name.startswith('_') and not name.startswith('get_') and name not in exclude
        ]

    @staticmethod
    def gaussian(in_array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        from vigra.filters import gaussianSmoothing
        if in_array.dtype == 'uint16':
            in_array = in_array.astype('float32')
            return gaussianSmoothing(in_array, sigma=sigma).astype('uint16')

        return gaussianSmoothing(in_array, sigma=sigma)

    @staticmethod
    def gaussian_gradient_magnitude(in_array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        from vigra.filters import gaussianGradientMagnitude
        return gaussianGradientMagnitude(in_array.astype('float32'), sigma=sigma)

    @staticmethod
    def clahe(
            in_array: np.ndarray,
            clip_limit: float = 3.0,
            tile_grid_size: tuple[int, int] = (127, 127),
            # cast_dtype: str = None,
            invert_output: bool = False,
            # gaussian_sigma: float = 0.0,
            auto_mask: bool = False,
            background_to_mean: bool = False,
            tile_grid_in_pixels: bool = False
    ) -> np.ndarray:
        from squirrel.library.normalization import clahe_on_image
        return clahe_on_image(
            in_array,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            # cast_dtype=cast_dtype,
            invert_output=invert_output,
            # gaussian_sigma=gaussian_sigma,
            auto_mask=auto_mask,
            background_to_mean=background_to_mean,
            tile_grid_in_pixels=tile_grid_in_pixels
        )

    @staticmethod
    def vsnr(
            in_array: np.ndarray,
            filters: list[dict] = None,
            is_gpu: bool = True,
            maxit: int = 20,
            algo: str = 'auto'
    ) -> np.ndarray:
        from squirrel.library.vsnr import vsnr_on_image
        return vsnr_on_image(
            in_array,
            filters=filters,
            is_gpu=is_gpu,
            maxit=maxit,
            algo=algo
        )

    @staticmethod
    def fft_highpass(
            in_array: np.ndarray,
            sigma: (float, float) = (10, 10),
            keep_zeros: bool = False,
            cast_dtype: str = None
    ) -> np.ndarray:
        return fft_highpass(in_array, sigma, keep_zeros=keep_zeros, cast_dtype=cast_dtype)


if __name__ == '__main__':

    # img = np.random.randint(0, 128, (200, 200), dtype=np.uint8)
    # imf = ImageFilter(img)
    # # imf = ImageFilter(np.zeros([10, 100, 100]))
    # print(imf.get_available_filter_names())
    # # imf.get_filtered(['gaussian', 'gaussian_gradient_magnitude'], [dict(sigma=1.0), dict(sigma=1.0)])
    # # imf.get_filtered_batch(['gaussian'], [dict(sigma=1.0)], n_workers=10)
    # result = imf.get_filtered(['clahe'], [dict(tile_grid_in_pixels=True, tile_grid_size=(31, 31))])
    #
    # from matplotlib import pyplot as plt
    # plt.imshow(np.concatenate([img, result], axis=1), cmap='gray')
    # plt.show()

    from squirrel.library.io import read_tif_slice
    img = read_tif_slice(filepath='/mnt/icem/hennies/tmp/clahe_test/image.tif', return_filepath=False)

    imf = ImageFilter(img)

    result = imf.get_filtered(
        [
            ['vsnr', dict(is_gpu=True, maxit=100, filters=[
                dict(name='Gabor', sigma=[2, 35], theta=0, noise_level=0.5),  #, frequency=0.3),
            ])],
            ['vsnr', dict(is_gpu=True, maxit=100, filters=[
                dict(name='Gabor', sigma=[2, 35], theta=90, noise_level=0.5),
            ])],
            ['vsnr', dict(is_gpu=True, maxit=100, filters=[
                dict(name='Gabor', sigma=[5, 35], theta=90, noise_level=0.5),
            ])],
            ['clahe', dict(tile_grid_in_pixels=True, tile_grid_size=(63, 63))],
            ['gaussian', dict(sigma=1.5)]
        ]
    )

    from matplotlib import pyplot as plt
    plt.imshow(np.concatenate([img, result], axis=1), cmap='gray')
    plt.show()

