
import numpy as np


def str2values(input_str, dtype='float'):
    import re
    if dtype == 'float':
        return np.array([float(x) for x in (re.findall('\d+\.\d+|\d+', input_str))])
    if dtype == 'int':
        return np.array([int(x) for x in (re.findall('\d+', input_str))])


def parse_kwargs_list(kwargs_list):
    """
    Convert a list of strings of the form "key:value" into a dictionary,
    automatically casting values to int, float, or keeping as string.

    Args:
        kwargs_list (list[str]): List of strings like ["key1:val1", "key2:val2"]

    Returns:
        dict: Dictionary with proper types.
    """
    result = {}
    for item in kwargs_list:
        try:
            key, value = item.split(":", 1)  # split only on first colon
        except ValueError:
            raise ValueError(f"Invalid syntax for '{item}'. Expected 'key:value'")
        key = key.strip()
        value = value.strip()

        # Try to convert to int
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            value = int(value)
        else:
            # Try to convert to float
            try:
                value = float(value)
            except ValueError:
                pass  # keep as string if not int/float

        result[key] = value
    return result


if __name__ == '__main__':

    print(str2values('1,2,3,4,5', dtype='int'))
    print(str2values('1,2,3,4,5', dtype='float'))
    print(str2values('1.5,2.4,3.3,4.2,5.1', dtype='float'))
    print(str2values('1.5,2.4,3.3,4.2,5.1', dtype='int'))
    print(str2values('1.5,2,3.3,4,5.1', dtype='float'))
    print(str2values('1.5,2,3.3,4,5.1', dtype='int'))
    """
    Correct output: 
    [1 2 3 4 5]
    [1. 2. 3. 4. 5.]
    [1.5 2.4 3.3 4.2 5.1]
    [1 5 2 4 3 3 4 2 5 1]
    [1.5 2.  3.3 4.  5.1]
    [1 5 2 3 3 4 5 1]
    """

