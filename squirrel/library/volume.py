
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
