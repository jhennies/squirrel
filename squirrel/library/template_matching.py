
from skimage.feature import match_template
import numpy as np


def match_template_on_image(image, template):

    from ..library.transformation import setup_translation_matrix
    from ..library.affine_matrices import AffineMatrix
    print(f'image.shape = {image.shape}')
    result = match_template(image, template)
    print(f'result.shape = {result.shape}')
    y, x = np.unravel_index(np.argmax(result), result.shape)
    print(f'x = {x}')
    print(f'y = {y}')

    return AffineMatrix(
        parameters=setup_translation_matrix([y, x], ndim=2).flatten(),
        pivot=[0., 0.]
    )

