import unittest
import warnings
import numpy as np


class TestImageFilter(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_gaussian(self):
        print(f'Testing Gaussian filter ...')
        from squirrel.library.filters import ImageFilter
        from vigra.filters import gaussianSmoothing
        from skimage.data import camera
        img = camera()

        gaussian_ref = gaussianSmoothing(img, 2)
        imf = ImageFilter(img)
        gaussian = imf.get_filtered([['gaussian', dict(sigma=2)]])

        np.testing.assert_equal(gaussian_ref, gaussian)

    def test_gaussian_gradient_magnitude(self):
        print(f'Testing Gaussian gradient magnitude ...')
        from squirrel.library.filters import ImageFilter
        from vigra.filters import gaussianGradientMagnitude
        from skimage.data import camera
        img = camera()

        gaussian_ref = gaussianGradientMagnitude(img.astype('float32'), 2)
        imf = ImageFilter(img)
        gaussian = imf.get_filtered([['gaussian_gradient_magnitude', dict(sigma=2)]])

        np.testing.assert_equal(gaussian_ref, gaussian)

    def test_clahe(self):
        print(f'Testing CLAHE filter ...')
        from squirrel.library.filters import ImageFilter
        from cv2 import createCLAHE
        from skimage.data import camera
        img = camera()

        clip_limit = 3.0
        tile_grid_size = (127, 127)
        clahe = createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        clahe_filtered_ref = clahe.apply(img)
        imf = ImageFilter(img)
        clahe_filtered = imf.get_filtered([['clahe', dict()]])

        np.testing.assert_equal(clahe_filtered_ref, clahe_filtered)

    def test_vsnr(self):
        print(f'Testing VSNR ...')
        from squirrel.library.filters import ImageFilter
        from pyvsnr import vsnr2d
        from skimage.data import camera
        img = camera()

        filters = [dict(name='Gabor', noise_level=0.35, sigma=[1, 50], theta=90)]
        vsnr_ref = vsnr2d(img, filters)
        imf = ImageFilter(img)
        vsnr = imf.get_filtered([['vsnr', dict(filters=dict(name='Gabor', noise_level=0.35, sigma=[1, 50], theta=90))]])

        np.testing.assert_equal(vsnr_ref, vsnr)
