
import numpy as np


def inverse_of_sequence(transforms):

    result_transforms = []
    for transform in transforms:
        result_transforms.append(np.linalg.inv(transform))

    return result_transforms


def dot_product_of_sequences(transforms_a, transforms_b, inverse=(0, 0)):

    transforms_a = np.array(transforms_a)
    transforms_b = np.array(transforms_b)

    n_transforms = min(len(transforms_a), len(transforms_b))

    if inverse[0]:
        transforms_a = inverse_of_sequence(transforms_a)
    if inverse[1]:
        transforms_b = inverse_of_sequence(transforms_b)

    result_transforms = []
    for idx in range(n_transforms):
        transform_a = transforms_a[idx]
        transform_b = transforms_b[idx]
        result_transforms.append(np.dot(transform_a, transform_b))

    return result_transforms
