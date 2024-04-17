
import numpy as np


def load_affine_stack_from_multiple_files(filepaths):

    stack = AffineStack(filepath=filepaths[0])
    for filepath in filepaths[1:]:
        stack.append(AffineStack(filepath=filepath))

    return stack


class AffineStack:

    def __init__(
            self,
            stack=None,
            filepath=None,
            is_sequenced=None,
            pivot=None
    ):

        self._filepath = None
        self._stack = None
        self._ndim = None
        self.is_sequenced = None
        self._it = 0
        self._pivot = None
        self._meta = dict()
        if stack is not None:
            assert filepath is None
            self.set_from_stack(stack, is_sequenced=is_sequenced, pivot=pivot)
            return
        if filepath is not None:
            assert stack is None
            self.set_from_file(filepath, is_sequenced=is_sequenced, pivot=pivot)
            return
        self.is_sequenced = is_sequenced
        self.set_pivot(pivot)

    def _validate_stack(self):
        pivots = [m.get_pivot() for m in self[:]]
        if not np.array((x == pivots[0]).all() for x in pivots).all():
            print(f'All pivots of stack must be equal!')
            return False
        if not (self._pivot == pivots[0]).all():
            print('Pivots in stack must match stack pivot')
            return False
        if len(self._pivot) != self._ndim:
            print(f'Pivot shape must match number of dimensions! pivot={self._pivot}; ndim={self._ndim}')
            return False
        ndims = np.array([m.get_ndim() for m in self[:]])
        if not np.array(x == ndims for x in ndims).all():
            print('All ndims of stack must be equal!')
            return False
        if self._ndim != ndims[0]:
            print(f'ndims in stack must match stack ndim! {ndims[0]} != {self._ndim}')
            return False
        return True

    def set_from_stack(self, stack, is_sequenced=None, pivot=None):
        if not isinstance(stack[0], AffineMatrix):
            stack = [AffineMatrix(parameters=parameters, pivot=pivot) for parameters in stack]
        self._stack = stack
        self._set_ndim()
        self.set_pivot(pivot)
        if self._validate_stack():
            self.is_sequenced = is_sequenced
            return
        raise RuntimeError('Validation of stack failed!')

    def update_stack(self, stack):
        self.set_from_stack(stack, is_sequenced=self.is_sequenced, pivot=self.get_pivot())

    def set_from_file(self, filepath, is_sequenced=None, pivot=None):
        import json
        with open(filepath, mode='r') as f:
            stack_data = json.load(f)

        stack = stack_data['transforms']
        if is_sequenced is None:
            is_sequenced = stack_data['sequenced']
        if pivot is None:
            pivot = stack_data['pivot']

        if 'meta' in stack_data:
            self._meta = stack_data['meta']

        self.set_from_stack(stack, is_sequenced=is_sequenced, pivot=pivot)

    def to_file(self, filepath):
        import json
        out_data = dict(
            transforms=self['C', :].tolist(),
            sequenced=self.is_sequenced,
            pivot=self.get_pivot().tolist(),
            meta=self._meta
        )
        try:
            with open(filepath, 'w') as f:
                json.dump(out_data, f, indent=2)
        except TypeError:
            print(out_data)
            raise

    def set_pivot(self, pivot=None):
        if pivot is None and self._stack is not None:
            pivots = [m.get_pivot() for m in self[:]]
            assert np.array((x == np.array(pivots[0])).all() for x in pivots).all(), 'Not all pivots are equal!'
            self._pivot = np.array(pivots[0])
            return
        if pivot is None and self._stack is None:
            self._pivot = np.array([0.] * self._ndim)
            return
        if pivot is not None and self._stack is not None:
            assert len(pivot) == self._ndim
            self._pivot = np.array(pivot, dtype=float)
            return
        if pivot is not None and self._stack is None:
            self._pivot = np.array(pivot, dtype=float)
            self._ndim = len(pivot)

    def _set_ndim(self):
        ndims = np.array([m.get_ndim() for m in self[:]])
        if (ndims == 2).all():
            self._ndim = 2
            return
        if (ndims == 3).all():
            self._ndim = 3

    def get_pivot(self):
        return self._pivot

    def _append_meta(self, other):
        for k, v in self._meta.items():
            if k in other._meta:
                if type(self._meta[k]) == list:
                    assert type(other._meta[k]) == list
                    self._meta[k].extend(other._meta[k])

    def append(self, other):
        if isinstance(other, AffineStack):
            assert self.is_sequenced == other.is_sequenced
            assert (self.get_pivot() == other.get_pivot()).all(), f'{self.get_pivot()} != {other.get_pivot()}'
            self.set_from_stack(self[:] + other[:], is_sequenced=self.is_sequenced, pivot=self.get_pivot())
            self._append_meta(other)
            return
        if isinstance(other, AffineMatrix):
            assert (self.get_pivot() == other.get_pivot()).all(), f'{self.get_pivot()} != {other.get_pivot()}'
            self.set_from_stack(self[:] + [other], is_sequenced=self.is_sequenced, pivot=self.get_pivot())

    def get_ndim(self):
        return self._ndim

    def __getitem__(self, item):
        """
        :param item:
        as[i]       -> AffineMatrix object
        as['Ms', i] -> [[a, b, ty], [c, d, tx], [0, 0, 1]]
        as['M', i]  -> [[a, b, ty], [c, d, tx]]
        as['Cs', i] -> [a, b, ty, c, d, tx, 0, 0, 1]
        as['C', i]  -> [a, b, ty, c, d, tx]
        as[:]       -> AffineStack[AffineMatrixA, AffineMatrixB, ...]
        as['Ms', :] -> [[[a, b, ty], [c, d, tx], [0, 0, 1]], ...]
        :return:
        """
        if isinstance(item, int):
            return self._stack[item]
        if isinstance(item, tuple):
            assert isinstance(item[0], str)
            items = self._stack[item[1]]
            if isinstance(items, AffineMatrix):
                return items.get_matrix(order=item[0])
            return np.array([x.get_matrix(order=item[0]) for x in items])
        if isinstance(item, slice):
            if self._stack is not None:
                return self._stack[item]
            else:
                return []
        raise ValueError('Invalid indexing!')

    def __setitem__(self, key, value):
        self._stack[key] = value

    def __neg__(self):
        return AffineStack(
            stack=[-x for x in self[:]],
            is_sequenced=self.is_sequenced
        )

    def __len__(self):
        return len(self._stack)

    def __iter__(self):
        self._it = 0
        return self

    def __next__(self):
        if self._it < len(self):
            x = self[self._it]
            self._it += 1
            return x
        else:
            raise StopIteration

    def __mul__(self, other):
        return AffineStack(
            stack=[self[idx] * other[idx] for idx in range(len(self))],
            is_sequenced=self.is_sequenced
        )

    def _new_stack_with_same_meta(self, new_stack):
        ns = AffineStack(stack=new_stack, is_sequenced=self.is_sequenced, pivot=self.get_pivot())
        ns._meta = self._meta
        return ns

    def copy(self):
        return self._new_stack_with_same_meta(self._stack.copy())

    def get_smoothed_stack(self, sigma):
        from scipy.ndimage import gaussian_filter1d
        return self._new_stack_with_same_meta(gaussian_filter1d(self['C', :], sigma, axis=0))

    def get_sequenced_stack(self):

        assert not self.is_sequenced, 'A sequenced stack cannot be sequenced again!'

        stack = []
        for idx, matrix in enumerate(self[:]):
            if idx == 0:
                stack.append(self[0])
                continue
            stack.append(stack[idx - 1] * self[idx])

        out_stack = self._new_stack_with_same_meta(stack)
        out_stack.is_sequenced = True
        return out_stack

    def get_scaled(self, scale):

        assert self.is_sequenced, 'Scaling only works for sequenced stacks!'

        scaled_stack = [item.get_scaled(scale).get_matrix('C') for item in self]

        if scale > 1 or 1/scale - int(1/scale) != 0:
            # z-interpolation to extend the stack
            from scipy.ndimage import zoom
            scaled_stack = zoom(
                scaled_stack, (scale, 1.),
                order=1, grid_mode=True, mode='grid-constant'
            )
        else:
            scaled_stack = [
                scaled_stack[idx]
                for idx in range(0, len(scaled_stack), int(1/scale))
            ]

        return self._new_stack_with_same_meta(scaled_stack)

    def set_meta(self, name, data):
        if type(data) == np.ndarray:
            data = data.tolist()
        self._meta[name] = data

    def get_meta(self, name):
        return self._meta[name]

    def exists_meta(self, name):
        if name in self._meta.keys():
            return True
        return False


class AffineMatrix:

    def __init__(
            self,
            parameters=None,
            elastix_parameters=None,
            filepath=None,
            pivot=None
    ):
        self._parameters = None
        self._ndim = None
        self._pivot = None
        if parameters is not None:
            self.set_from_parameters(parameters, pivot=pivot)
        if elastix_parameters is not None:
            self.set_from_elastix(elastix_parameters)
        if filepath is not None:
            self.set_from_file(filepath)

    def _validate_parameters(self, parameters):
        parameters = np.array(parameters)
        if len(parameters) == 6:
            self._ndim = 2
            return True
        if len(parameters) == 12:
            self._ndim = 3
            return True
        return False

    def set_from_parameters(self, parameters, pivot=None):
        if self._validate_parameters(parameters):
            self._parameters = np.array(parameters, dtype=float)
            self.set_pivot(pivot)
            return
        raise RuntimeError(f'Validation of parameters failed! {parameters}')

    def update_parameters(self, parameters):
        self.set_from_parameters(parameters, pivot=self.get_pivot())

    def set_from_elastix(self, parameters):
        assert isinstance(parameters[0], str), \
            'Elastix parameters must be in the format: ["transform", [parameters, ...]]'
        assert parameters[0] in ['translation', 'rigid', 'SimilarityTransform', 'affine']
        from ..library.elastix import elastix_to_c
        parameters = elastix_to_c(*parameters)
        self.set_from_parameters(parameters)

    def get_matrix(self, order='C'):

        if order[0] == 'C':
            if len(order) == 2 and order[1] == 's':
                return np.concatenate((self._parameters.copy(), [0.] * self._ndim + [1.]), axis=0)
            return self._parameters.copy()
        if order[0] == 'M':
            out_matrix = np.reshape(self._parameters.copy(), (self._ndim, self._ndim + 1), order='C')
            if len(order) == 2 and order[1] == 's':
                return np.concatenate((out_matrix, [[0.] * self._ndim + [1.]]), axis=0)
            return out_matrix

    def set_from_file(self, filepath):
        from squirrel.library.io import get_filetype
        filetype = get_filetype(filepath)

        if filetype == 'json':
            import json
            with open(filepath, mode='r') as f:
                matrix_data = json.load(f)
                self.set_from_parameters(parameters=matrix_data['transform'], pivot=matrix_data['pivot'])
            return

        if filetype == 'csv':
            from numpy import genfromtxt
            self.set_from_parameters(genfromtxt(filepath, delimiter=','))
            raise NotImplementedError
            return

        raise ValueError(f'Invalid filetype: {filetype}')

    def set_pivot(self, pivot):
        self._pivot = np.array(pivot, dtype=float) if pivot is not None else np.array([0.] * self._ndim)
        assert len(self._pivot) == self._ndim

    def get_pivot(self):
        return self._pivot

    def to_file(self, filepath):
        import json
        out_data = dict(
            transform=self.get_matrix(order='C').tolist(),
            pivot=self._pivot.tolist()
        )
        with open(filepath, mode='w') as f:
            json.dump(out_data, f, indent=2)

    def _ms_to_c(self, matrix):
        if self._ndim == 2:
            assert (matrix[2] == [0., 0., 1.]).all()
            return matrix[:2].flatten()
        if self._ndim == 3:
            assert (matrix[3] == [0., 0., 0., 1]).all()
            return matrix[:3].flatten()

    def get_ndim(self):
        return self._ndim

    def inverse(self):
        inv = np.linalg.inv(self.get_matrix(order='Ms'))
        inv = self._ms_to_c(inv)
        return AffineMatrix(parameters=inv)

    def dot(self, other):
        assert type(other) == AffineMatrix
        result = np.dot(self.get_matrix(order='Ms'), other.get_matrix(order='Ms'))
        result = self._ms_to_c(result)
        return AffineMatrix(parameters=result)

    def __neg__(self):
        return self.inverse()

    def __mul__(self, other):
        return self.dot(other)

    def copy(self):
        return AffineMatrix(self.get_matrix(), pivot=self.get_pivot())

    def get_translation(self):
        return self.get_matrix('M')[:self._ndim, self._ndim]

    def set_translation(self, translation):
        assert len(translation) == self._ndim
        matrix = self.get_matrix('M')
        matrix[:self._ndim, self._ndim] = translation
        self.set_from_parameters(matrix.flatten(), pivot=self.get_pivot())

    def get_scaled(self, scale):
        matrix = self.copy()
        matrix.set_translation(matrix.get_translation() * scale)
        pivot_matrix = AffineMatrix([1., 0., matrix.get_pivot()[0], 0., 1., matrix.get_pivot()[1]])
        matrix = matrix * pivot_matrix
        pivot_matrix.set_translation(pivot_matrix.get_translation() * scale)
        return (-pivot_matrix) * matrix

    def decompose(self):
        from transforms3d.affines import decompose
        from squirrel.library.transformation import (
            setup_translation_matrix,
            setup_scale_matrix,
            setup_shear_matrix
        )
        t, r, z, s = decompose(self.get_matrix('Ms'))
        r_ = np.zeros([self._ndim, self._ndim + 1], dtype=float)
        r_[:r.shape[0], :r.shape[1]] = r
        return (
            AffineMatrix(setup_translation_matrix(t, ndim=self.get_ndim()).flatten()),
            AffineMatrix(r_.flatten()),
            AffineMatrix(setup_scale_matrix(z, ndim=self.get_ndim()).flatten()),
            AffineMatrix(setup_shear_matrix(s, ndim=self.get_ndim()).flatten())
        )

    def shift_pivot_to_origin(self):
        matrix = self.get_matrix('Ms')
        pivot = self.get_pivot()
        offset = pivot - np.dot(matrix[:2, :2], pivot)
        pivot_matrix = np.array([
            [1., 0., offset[0]],
            [0., 1., offset[1]],
            [0., 0., 1.]
        ])
        matrix = np.dot(pivot_matrix, matrix)[:2]
        self.set_from_parameters(matrix.flatten(), pivot=[0., 0.])


if __name__ == '__main__':

    # TODO: the stuff here should be implemented as unittests

    if True:
        print('Testing the stack object -----------------')
        stk = AffineStack(
            stack=[[1.1, 0., 5, 0., 1.3, 2], [0.8, 0., -3, 0., 0.75, 1.5]],
            is_sequenced=False
        )
        # print(f'stk.get_ndim = {stk.get_ndim()}')
        # print(f'stk[0] = {stk[0]}')
        # print(f'stk[0].get_matrix() = {stk[0].get_matrix()}')
        # print(f'stk[1].get_matrix() = {stk[1].get_matrix()}')
        # print(f'stk["Ms", 0] = {stk["Ms", 0]}')
        # print(f'stk[:] = {stk[:]}')
        # print(f'stk["Ms", :] = {stk["Ms", :]}')
        # print(f'\nStack IO ---------------------')
        stk.set_meta('test', np.array([1, 2, 4]))
        fp = '/media/julian/Data/tmp/affine_stack_test.json'
        stk.to_file(fp)
        stk_loaded = AffineStack(filepath=fp)
        print(f'stk_loaded["Ms", :] = {stk_loaded["Ms", :]}')
        print(f'stk_loaded.is_sequenced = {stk_loaded.is_sequenced}')
        print(f'\nMathematical operations ---------------------')
        print(f'(-stk)["Ms", :] = {(-stk)["Ms", :]}')
        print(f'(stk * stk)["Ms", :] = {(stk * stk)["Ms", :]}')
        print(f'(stk * -stk)["Ms", :] = {(stk * -stk)["Ms", :]}')
        stk.append(-stk_loaded)
        print(f'Appended: stk["C", :] = {stk["C", :]}')
        # print(f'stk.smooth(2)["Ms", :] = {stk.get_smoothed_stack(2)["Ms", :]}')
        # print(f'before: stk["Ms", :] = {stk["C", :]}')
        # print(f'sequenced: stk["Ms", :] = {stk.get_sequenced_stack()["C", :]}')
        stk.is_sequenced = True
        print(f'scaled: {stk.get_scaled(2)["C", :]}')
        print(f'scaled: {stk.get_scaled(0.5)["C", :]}')


    if True:
        print('Testing the matrix object')
        am = AffineMatrix(parameters=[1.1, 0., 5, 0., 1.3, 2])
        # print(am.get_matrix(order='C'))
        # print(am.get_matrix(order='Cs'))
        # print(am.get_matrix(order='M'))
        # print(am.get_matrix(order='Ms'))
        # print((-am).get_matrix(order='Ms'))
        # print((am * am).get_matrix(order='Ms'))
        # print((am * -am).get_matrix(order='Ms'))
        # print((-am * am).get_matrix(order='Ms'))
        # print(am.get_matrix(order='Ms'))
        # print('\nTesting IO')
        # fp = '/media/julian/Data/tmp/affine_matrix_test.json'
        # am.to_file(fp)
        # am_loaded = AffineMatrix()
        # am_loaded.set_from_file(fp)
        # print(am_loaded.get_matrix(order='Ms'))
        # am3d = AffineMatrix(parameters=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], pivot=[100, 0, 0])
        # am3d.to_file('/media/julian/Data/tmp/affine_matrix_test_3d.json')

        print(f'non-scaled: {am.get_matrix("Ms")}')
        print(f'scaled: {am.get_scaled(2).get_matrix("Ms")}')
        print(f'non-scaled: {am.get_matrix("Ms")}')
        # from squirrel.library.transformation import scale_affine_matrix
        # print(scale_affine_matrix(am.get_matrix('Ms'), 2, [0., 0.]))

        print(f'decomposed:')
        decomp = am.decompose()
        for d in decomp:
            print(d.get_matrix())

