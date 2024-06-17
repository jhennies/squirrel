
import numpy as np


def list2roi(input_list, format='zyxdhw'):

    if format == 'zyxdhw':
        return np.s_[
            input_list[0]: input_list[0] + input_list[3] if input_list[3] != 0 else None,
            input_list[1]: input_list[1] + input_list[4] if input_list[4] != 0 else None,
            input_list[2]: input_list[2] + input_list[5] if input_list[5] != 0 else None,
        ]

    raise ValueError(f'Invalid roi format: {format}')


if __name__ == '__main__':

    print(list2roi([0, 1, 2, 3, 4, 5]))
    print(list2roi([3, 4, 5, 1, 0, 1]))
    """
    Expected output:
    (slice(0, 3, None), slice(1, 5, None), slice(2, 7, None))
    (slice(3, 4, None), slice(4, None, None), slice(5, 6, None))
    """
    a = np.sum(np.mgrid[:6, :6, :6], axis=0)
    print(a[list2roi([1, 2, 3, 0, 2, 1])].shape)
    """
    Expected output:
    (5, 2, 1)
    """
