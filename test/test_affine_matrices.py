import unittest
import warnings
import numpy as np

from squirrel.library.affine_matrices import AffineMatrix


class TestAffineMatrix(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_inverse(self):
        print(f'Testing AffineMatrix: inverse ...')
        params = np.array([
            [0.3, -0.02, 1],
            [0.12, 1.2, 2]
        ])
        A = AffineMatrix(parameters=params)
        B = A.inverse()

        I = (A * B).get_matrix('Ms')
        self.assertTrue(np.allclose(I, np.eye(3), atol=1e-6))

    def test_dot(self):
        print(f'Testing AffineMatrix: dot ...')
        params_a = np.array([
            [1, 0, 0],
            [0.12, 1.2, 2]
        ])
        params_b = np.array([
            [0.3, -0.02, 1],
            [0, 1, 0]
        ])
        A = AffineMatrix(parameters=params_a)
        B = AffineMatrix(parameters=params_b)
        C = A * B
        t = C.get_matrix('M')
        out = np.array([
            [0.3, -0.02, 1],
            [0.036, 1.1976, 2.12]
        ])
        self.assertTrue(np.allclose(t, out))

    def test_get_translation(self):
        print(f'Testing AffineMatrix: get_translation ...')
        t = [3, -2]
        A = AffineMatrix(translation=t)
        self.assertTrue(np.allclose(A.get_translation(), t))

    def test_set_translation(self):
        print(f'Testing AffineMatrix: set_translation ...')
        A = AffineMatrix(translation=[0, 0])
        A.set_translation([5, -3])
        self.assertTrue(np.allclose(A.get_translation(), [5, -3]))

    def test_get_scaled(self):
        print(f'Testing AffineMatrix: get_scaled ...')
        params = np.array([
            [0.3, -0.02, 1],
            [0.12, 1.2, 2]
        ])
        A = AffineMatrix(parameters=params, pivot=[10, 10])
        B = A.get_scaled(0.5)

        out = np.array([
            [0.3, -0.02, -1.7],
            [0.12,  1.2,  9.2]
        ])

        self.assertTrue(np.allclose(B.get_matrix('M'), out))

    def test_decompose(self):
        # translation + scale (no rotation to keep it simple)
        T = AffineMatrix(translation=[2, 3])
        S = AffineMatrix(parameters=[[2, 0, 0],
                                     [0, 3, 0]])

        A = T * S

        t, r, z, s = A.decompose()

        self.assertTrue(np.allclose(t.get_translation(), [2, 3]))
        self.assertTrue(np.allclose(z.get_matrix('M')[:2, :2], [[2, 0], [0, 3]], atol=1e-6))

    def test_shift_pivot_to_origin(self):
        A = AffineMatrix(translation=[1, 0], pivot=[2, 0])
        A.shift_pivot_to_origin()

        self.assertTrue(np.allclose(A.get_pivot(), [0, 0]))

    def test_return_3d(self):
        A = AffineMatrix(translation=[1, 2], pivot=[3, 4])
        B = A.return_3d(axis=2)

        self.assertEqual(B.get_ndim(), 3)

        t = B.get_translation()
        self.assertTrue(np.allclose(t[:2], [1, 2]))
        self.assertTrue(np.allclose(t[2], 0))

        pivot = B.get_pivot()
        self.assertTrue(np.allclose(pivot, [3, 4, 0]))
