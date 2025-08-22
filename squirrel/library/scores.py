import numpy as np


def intersection_over_union(im1, im2):

    union = np.zeros(im2.shape, dtype=bool)
    union[im2 > 0] = True
    union[im1 > 0] = True

    intersection = np.zeros(im2.shape, dtype=bool)
    intersection[np.logical_and(im2 > 0, im1 > 0)] = True

    union_sum = union.sum()

    if union_sum == 0:
        return 0

    iou = intersection.sum() / union_sum

    return iou
