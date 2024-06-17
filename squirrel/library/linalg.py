
import numpy as np


# NOTE: deprecated
# def inverse_of_sequence(transforms):
#
#     result_transforms = []
#     for transform in transforms:
#         result_transforms.append(np.linalg.inv(transform))
#
#     return result_transforms


# NOTE: deprecated
# def dot_product_of_sequences(transforms_a, transforms_b, inverse=(0, 0)):
#
#     transforms_a = np.array(transforms_a)
#     transforms_b = np.array(transforms_b)
#
#     n_transforms = min(len(transforms_a), len(transforms_b))
#
#     if inverse[0]:
#         transforms_a = inverse_of_sequence(transforms_a)
#     if inverse[1]:
#         transforms_b = inverse_of_sequence(transforms_b)
#
#     result_transforms = []
#     for idx in range(n_transforms):
#         transform_a = transforms_a[idx]
#         transform_b = transforms_b[idx]
#         result_transforms.append(np.dot(transform_a, transform_b))
#
#     return result_transforms


# NOTE: deprecated
# def modify_step_in_sequence(transforms, idx, affine, replace=False):
#
#     transforms = np.array(transforms)
#     assert transforms[idx].shape == (3, 3)
#
#     affine = list(affine)
#     affine.extend([0, 0, 1])
#     affine = np.reshape(affine, (3, 3))
#
#     if replace:
#         transforms[idx] = affine
#         return transforms
#
#     transforms[idx] = np.dot(transforms[idx], affine)
#     return transforms


# NOTE: deprecated
# def create_affine_sequence(length):
#
#     return [[1., 0., 0., 0., 1., 0.]] * length
