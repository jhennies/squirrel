
from skimage.feature import match_template
import numpy as np


def match_template_on_image(image, template):

    from ..library.transformation import setup_translation_matrix
    from ..library.elastix import save_transforms
    print(f'image.shape = {image.shape}')
    result = match_template(image, template)
    print(f'result.shape = {result.shape}')
    y, x = np.unravel_index(np.argmax(result), result.shape)
    print(f'x = {x}')
    print(f'y = {y}')

    return save_transforms(
        setup_translation_matrix([y, x], ndim=2),
        None,
        param_order='M',
        save_order='C',
        ndim=2
    )

