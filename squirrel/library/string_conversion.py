
import numpy as np


def str2values(input_str, dtype='float'):
    import re
    if dtype == 'float':
        return np.array([float(x) for x in (re.findall('\d+\.\d+|\d+', input_str))])
    if dtype == 'int':
        return np.array([int(x) for x in (re.findall('\d+', input_str))])


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

