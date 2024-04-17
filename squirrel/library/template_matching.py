
from skimage.feature import match_template
import numpy as np


def match_template_on_image(image, template):

    from ..library.transformation import setup_translation_matrix
    from ..library.affine_matrices import AffineMatrix
    # print(f'image.shape = {image.shape}')
    result = match_template(image, template)
    # print(f'result.shape = {result.shape}')
    y, x = np.unravel_index(np.argmax(result), result.shape)
    # print(f'x = {x}')
    # print(f'y = {y}')

    return AffineMatrix(
        parameters=setup_translation_matrix([y, x], ndim=2).flatten(),
        pivot=[0., 0.]
    )


def match_template_on_stack_slice(
        stack_handle,
        idx,
        search_roi,
        template,
        determine_bounds=False
):

    from ..library.io import crop_roi

    if search_roi is None:
        z_slice = stack_handle[idx]
    else:
        z_slice, _ = crop_roi(stack_handle, search_roi + [idx])

    transform = match_template_on_image(
        z_slice,
        template
    )

    if determine_bounds:
        from ..library.image import get_bounds
        return transform, get_bounds(z_slice, return_ints=True)

    return transform

