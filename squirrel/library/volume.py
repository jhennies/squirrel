
import numpy as np


def pad_volume(vol, min_shape, axes=None):

    if np.all(np.array(vol.shape) > np.array(min_shape)):
        return vol

    if axes is None:

        new_shape = np.max([vol.shape, min_shape], axis=0)

    else:
        new_shape = np.array(vol.shape)
        for a in axes:
            if vol.shape[a] < min_shape[a]:
                new_shape[a] = min_shape[a]

    t_vol = np.zeros(new_shape, dtype=vol.dtype)
    t_vol[:vol.shape[0], :vol.shape[1], :vol.shape[2]] = vol

    return t_vol


def _average(a, b):
    return np.add(a.astype(float), b.astype(float)) / 2


def _get_math_operation(operation):
    if operation == 'add':
        return np.add
    if operation == 'subtract':
        return np.subtract,
    if operation == 'multiply':
        return np.multiply
    if operation == 'divide':
        return np.divide
    if operation == 'min':
        return np.minimum
    if operation == 'max':
        return np.maximum
    if operation == 'average':
        return _average
    raise ValueError(f'Invalid value for operation: {operation}')


def _slice_calculator(stack_a, stack_b, idx, func, target_dtype):
    print(f'idx = {idx}')
    if target_dtype is not None:
        return func(stack_a[idx], stack_b[idx]).astype(target_dtype)
    return func(stack_a[idx], stack_b[idx])


def stack_calculator(
        stack_a, stack_b, operation='add', target_dtype=None, n_workers=1, verbose=False
):

    if verbose:
        print(f'Running stack calculator with operation = {operation}')
    func = _get_math_operation(operation)

    if n_workers == 1:
        return [_slice_calculator(stack_a, stack_b, idx, func) for idx in range(len(stack_a))]

    else:

        from multiprocessing import Pool
        with Pool(processes=n_workers) as p:
            tasks = [
                p.apply_async(_slice_calculator, (
                    stack_a, stack_b, idx, func, target_dtype
                ))
                for idx in range(len(stack_a))
            ]
            return [task.get() for task in tasks]


def running_volume_average(
        stack, axis=0, average_method='mean', window_size=None, operation=None, verbose=False
):
    assert average_method in ['mean', 'median']

    stack = np.array(stack)
    stack = stack.swapaxes(axis, 0)

    shape = stack.shape

    result = []

    if average_method == 'mean':

        assert window_size is None, 'average_method="mean" only implemented for window_size=None'

        cumulative = np.zeros((shape[1], shape[2]), dtype=float)

        for idx, slice in enumerate(stack):

            if verbose:
                print(f'idx = {idx} / {shape[0]}')

            cumulative = cumulative + slice
            cumulative_avg = cumulative / (idx + 1)

            result.append(cumulative_avg)

    if average_method == 'median':

        def _run_median_filter(idx):
            if verbose:
                print(f'idx = {idx} / {shape[0]}')

            if window_size is None:
                start = 0
            if window_size is not None:
                start = idx - window_size
                if start < 0:
                    start = 0

            return np.median(stack[start: idx], axis=0)

        n_workers = 16
        if n_workers == 1:
            for idx, slice in enumerate(stack):

                result.append(
                    _run_median_filter(idx)
                )

        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n_workers) as tpe:

                tasks = [
                    tpe.submit(_run_median_filter, idx)
                    for idx, slice in enumerate(stack)
                ]

                result = [task.result() for task in tasks]

    if operation is None:
        result = np.array(result).swapaxes(0, axis)
        return result

    if operation == 'difference':
        result = stack.astype(float) - result
        result = np.array(result).swapaxes(0, axis)
        return result

    if operation == 'difference-clip':
        result = np.clip(
            stack.astype(float) - result + np.iinfo(stack.dtype).max / 2,
            np.iinfo(stack.dtype).min,
            np.iinfo(stack.dtype).max
        ).astype(stack.dtype)
        result = np.array(result).swapaxes(0, axis)
        return result

    raise ValueError(
        f'Invalid value for operation: {operation}; possible values = [None, "difference", "difference-clip"'
    )


def axis_median_filter(
        stack,
        median_radius=2,
        axis=0,
        operation=None,
        n_workers=1,
        verbose=False
):

    stack = np.array(stack)
    stack = stack.swapaxes(axis, 0)

    shape = stack.shape

    def _run_median_filter(idx):
        if verbose:
            print(f'idx = {idx} / {shape[0]}')

        start = idx - median_radius
        if start < 0:
            start = 0
        stop = idx + median_radius + 1
        if stop > shape[0]:
            stop = shape[0]
        return np.median(stack[start: stop], axis=0)

    if n_workers == 1:
        result = [_run_median_filter(idx) for idx, slice in enumerate(stack)]

    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as tpe:

            tasks = [
                tpe.submit(_run_median_filter, idx)
                for idx, slice in enumerate(stack)
            ]

            result = [task.result() for task in tasks]

    if operation is None:
        result = np.array(result).swapaxes(0, axis)
        return result

    if operation == 'difference':
        result = stack.astype(float) - result
        result = np.array(result).swapaxes(0, axis)
        return result

    if operation == 'difference-clip':
        result = np.clip(
            stack.astype(float) - result + np.iinfo(stack.dtype).max / 2,
            np.iinfo(stack.dtype).min,
            np.iinfo(stack.dtype).max
        ).astype(stack.dtype)
        result = np.array(result).swapaxes(0, axis)
        result[stack == 0] = 0
        return result

    raise ValueError(
        f'Invalid value for operation: {operation}; possible values = [None, "difference", "difference-clip"'
    )
