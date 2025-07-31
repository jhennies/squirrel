
import numpy as np
import inspect


class ImageFilter:

    def __init__(self, in_array):
        self._in_array = in_array

    def _check_filter_names(self, filter_names):
        available_filters = self.get_available_filter_names()
        for filter_name in filter_names:
            if filter_name not in available_filters:
                raise ValueError(f'{filter_name} not in available filters: {available_filters}')

    def get_filtered(
            self, filter_names: list[str], kwargs: list[dict[str, float]], in_array: np.ndarray = None
    ) -> np.ndarray:

        self._check_filter_names(filter_names)

        result_array = self._in_array.copy() if in_array is None else in_array
        for idx, filter_name in enumerate(filter_names):
            result_array = getattr(self, filter_name)(result_array, **kwargs[idx])

        return result_array

    def get_filtered_stack(
            self, filter_names: list[str], kwargs: list[dict[str, float]], n_workers: int = 1
    ) -> np.ndarray:
        if not (2 < self._in_array.ndim <= 4):
            raise RuntimeError(f'in_array.ndim = {self._in_array.ndim}. \n'
                               'in_array needs to have ndim=3 (stack of 2D images) or ndim=4 (stack of 3D images) '
                               'for stack processing.')

        if n_workers == 1:
            result_array = np.array([
                self.get_filtered(filter_names, kwargs, in_array=im_slice)
                for im_slice in self._in_array
            ])

        else:
            print(f'Running ImageFilter stack with {n_workers} CPUs')
            from multiprocessing import Pool
            with Pool(processes=n_workers) as p:
                tasks = [
                    p.apply_async(
                        self.get_filtered, (filter_names, kwargs), dict(in_array=im_slice)
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
        return gaussianSmoothing(in_array, sigma=sigma)

    @staticmethod
    def gaussian_gradient_magnitude(in_array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        from vigra.filters import gaussianGradientMagnitude
        return gaussianGradientMagnitude(in_array.astype('float32'), sigma=sigma)


if __name__ == '__main__':

    imf = ImageFilter(np.zeros([10, 100, 100]))
    print(imf.get_available_filter_names())
    # imf.get_filtered(['gaussian', 'gaussian_gradient_magnitude'], [dict(sigma=1.0), dict(sigma=1.0)])
    imf.get_filtered_batch(['gaussian'], [dict(sigma=1.0)], n_workers=10)
