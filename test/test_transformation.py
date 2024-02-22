import unittest
import warnings
import os
from shutil import rmtree
import numpy as np

from random import randint


class TestTransformation(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_smooth_2d_affine_sequence(self):

        print('Testing smooth_2d_affine_sequence ...')
        from squirrel.library.transformation import smooth_2d_affine_sequence

        sequence = [[0.999274, -0.000258773, 2.0483957015001595, -0.000441597, 0.999166, 2.8893408334999164]]

        sigma = 0.
        seq_smooth = smooth_2d_affine_sequence(sequence, sigma)

        assert np.allclose(sequence, seq_smooth, atol=1e-10), \
            'Without smoothing the input and output sequences must match! \n' \
            f'sequence = {sequence}\n' \
            f'seq_smooth = {seq_smooth}'
        print('... sigma = 0.0; Output is equal to input.')

        #
        # sigma = 1.
        # seq_smooth = smooth_2d_affine_sequence(sequence, sigma)
        # # TODO
        # print('... sigma = 1.0; Output as expected.')
