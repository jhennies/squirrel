
from skimage.feature import match_template
import numpy as np


def match_template_on_image(image, template):

    result = match_template(image, template)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    return x, y

