
import numpy as np


class AffineStack:

    def __init__(self, stack=None, filepath=None, is_sequenced=None):

        self._filepath = None
        self._stack = None
        self._ndim = None
        self.is_sequenced = None
        self._it = 0
        if stack is not None:
            assert filepath is None
            self.set_from_stack(stack, is_sequenced=is_sequenced)
        if filepath is not None:
            assert stack is None
            self.set_from_file(filepath, is_sequenced=is_sequenced)

    def _validate_stack(self, stack):
        ndims = np.array([m.get_ndim() for m in stack])
        if (ndims == 2).all():
            self._ndim = 2
            return True
        if (ndims == 3).all():
            self._ndim = 3
            return True
        return False

    def set_from_stack(self, stack, is_sequenced=None):
        if not isinstance(stack[0], AffineMatrix):
            stack = [AffineMatrix(parameters=parameters) for parameters in stack]
        if self._validate_stack(stack):
            self._stack = stack
            self.is_sequenced = is_sequenced
            return
        raise RuntimeError('Validation of stack failed!')

    def set_from_file(self, filepath, is_sequenced=None):
        import json
        with open(filepath, mode='r') as f:
            stack_data = json.load(f)

        stack = stack_data['transforms']
        if is_sequenced is None:
            is_sequenced = stack_data['sequenced']

        self.set_from_stack(stack, is_sequenced=is_sequenced)

    def to_file(self, filepath):
        import json
        out_data = dict(
            transforms=self['C', :].tolist(),
            sequenced=self.is_sequenced
        )
        with open(filepath, 'w') as f:
            json.dump(out_data, f, indent=2)
    
    def smooth(self, sigma):
        from scipy.ndimage import gaussian_filter1d
        return AffineStack(stack=gaussian_filter1d(self['C', :], sigma, axis=0))

    def append(self, other):
        assert self.is_sequenced == other.is_sequenced
        self.set_from_stack(self[:] + other[:])

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
        as[:]       -> [AffineMatrixA, AffineMatrixB, ...]
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
            return self._stack[item]
        raise ValueError('Invalid indexing!')

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


class AffineMatrix:

    def __init__(self, parameters=None, elastix_parameters=None, filepath=None):
        self._parameters = None
        self._ndim = None
        if parameters is not None:
            self.set_from_parameters(parameters)
        if elastix_parameters is not None:
            self.set_from_elastix(elastix_parameters)
        if filepath is not None:
            self.set_from_file(filepath)

    def _validate_parameters(self, parameters):
        if len(parameters) == 6:
            self._ndim = 2
            return True
        if len(parameters) == 12:
            self._ndim = 3
            return True
        return False

    def set_from_parameters(self, parameters):
        if self._validate_parameters(parameters):
            self._parameters = np.array(parameters, dtype=float)
            return
        raise RuntimeError(f'Validation of parameters failed! {parameters}')

    def set_from_elastix(self, parameters):
        parameters = self._elastix_to_c(parameters)
        self.set_from_parameters(parameters)

    def get_matrix(self, order='C'):

        if order[0] == 'C':
            if len(order) == 2 and order[1] == 's':
                return np.concatenate((self._parameters, [0., 0., 1.]), axis=0)
            return self._parameters
        if order[0] == 'M':
            out_matrix = np.reshape(self._parameters, (self._ndim, self._ndim + 1), order='C')
            if len(order) == 2 and order[1] == 's':
                return np.concatenate((out_matrix, [[0., 0., 1.]]), axis=0)
            return out_matrix

    def set_from_file(self, filepath):
        import json
        with open(filepath, mode='r') as f:
            self.set_from_parameters(json.load(f))

    def to_file(self, filepath):
        import json
        with open(filepath, mode='w') as f:
            json.dump(self.get_matrix(order='C').tolist(), f, indent=2)

    def _ms_to_c(self, matrix):
        if self._ndim == 2:
            assert (matrix[2] == [0., 0., 1.]).all()
            return matrix[:2].flatten()
        if self._ndim == 3:
            assert (matrix[3] == [0., 0., 0., 1]).all()
            return matrix[:3].flatten()

    def _elastix_to_c(self, parameters):
        raise NotImplementedError

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


if __name__ == '__main__':

    # TODO: the stuff here should be implemented as unittests

    if True:
        print('Testing the stack object -----------------')
        stk = AffineStack(
            stack=[[1.1, 0., 5, 0., 1.3, 2], [0.8, 0., -3, 0., 0.75, 1.5]],
            is_sequenced=False
        )
        print(f'stk.get_ndim = {stk.get_ndim()}')
        print(f'stk[0] = {stk[0]}')
        print(f'stk[0].get_matrix() = {stk[0].get_matrix()}')
        print(f'stk[1].get_matrix() = {stk[1].get_matrix()}')
        print(f'stk["Ms", 0] = {stk["Ms", 0]}')
        print(f'stk[:] = {stk[:]}')
        print(f'stk["Ms", :] = {stk["Ms", :]}')
        print(f'\nStack IO ---------------------')
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
        print(f'Appended: stk["Ms", :] = {stk["Ms", :]}')
        print(f'stk.smooth(2)["Ms", :] = {stk.smooth(2)["Ms", :]}')

    if False:
        print('Testing the matrix object')
        am = AffineMatrix(parameters=[1.1, 0., 5, 0., 1.3, 2])
        print(am.get_matrix(order='C'))
        print(am.get_matrix(order='Cs'))
        print(am.get_matrix(order='M'))
        print(am.get_matrix(order='Ms'))
        print((-am).get_matrix(order='Ms'))
        print((am * am).get_matrix(order='Ms'))
        print((am * -am).get_matrix(order='Ms'))
        print((-am * am).get_matrix(order='Ms'))
        print(am.get_matrix(order='Ms'))
        print('\nTesting IO')
        fp = '/media/julian/Data/tmp/affine_matrix_test.json'
        am.to_file(fp)
        am_loaded = AffineMatrix()
        am_loaded.set_from_file(fp)
        print(am_loaded.get_matrix(order='Ms'))

