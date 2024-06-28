
import numpy as np


def sift_log_line_to_affine(line):
    import re

    matrix = np.array([float(x) for x in (re.findall('[+-]*\d+\.\d+', line))])
    # The fiji matrix has inverted x and y axes
    matrix = [matrix[4], matrix[3], -matrix[5], matrix[1], matrix[0], -matrix[2]]

    from squirrel.library.affine_matrices import AffineMatrix
    return AffineMatrix(parameters=matrix)


def sift_log_to_affine_stack(
        log_lines
):

    from squirrel.library.affine_matrices import AffineStack

    transforms = AffineStack(pivot=[0., 0.], is_sequenced=False)
    for line in log_lines:
        if line.startswith('Transformation Matrix:'):
            transforms.append(sift_log_line_to_affine(line))

    return transforms


if __name__ == '__main__':

    log_filepath = ('/media/julian/Data/projects/concepcion/'
                    'cryo_fib_pre_processing/2024-02-27_JC_C3/fiji_sift/raw-translation-fiji.txt')
    f = open(log_filepath, mode='r')
    l = f.readlines()

    affines = sift_log_to_affine_stack(l)

    print(affines['C', :])
