
import numpy as np


class AffineSequence:

    def __init__(self, filepath=None, sequence=None, order='Cs'):

        self._filepath = None
        self._sequence = None
        self.is_sequenced = None
        if filepath is not None:
            assert sequence is None
            self.set_from_file(filepath)
        if sequence is not None:
            assert filepath is None
            self.set_from_sequence(sequence, order=order)

    def set_from_file(self, filepath):
        pass

    def set_from_sequence(self, sequence, order='Cs'):
        """

        :param sequence:
        :param order: ['C', 'Cs', 'M', 'Ms', 'elastix']
        :return:
        """
        pass

    def to_file(self, filepath, order='Cs'):
        pass

    def get_sequence(self, order='Cs'):
        pass
    
    def smooth(self, sigma):
        pass

    def __getitem__(self, item):
        """
        :param item:
        as[i]       -> [[a, b, ty], [c, d, tx], [0, 0, 1]]
        as['M', i]  -> [[a, b, ty], [c, d, tx], [0, 0, 1]]
        as['Ms', i] -> [[a, b, ty], [c, d, tx]]
        as['C', i]  -> [a, b, ty, c, d, tx, 0, 0, 1]
        as['Cs', i] -> [a, b, ty, c, d, tx]
        :return:
        """
        pass

    def __neg__(self):
        # TODO implement the inverse
        pass

    def __mul__(self, other):
        # TODO implement the dot product
        pass


class AffineMatrix:

    def __init__(self, parameters=None):
        self._parameters = None
        self._ndim = None
        if parameters is not None:
            self.set_from_parameters(parameters)

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
        raise NotImplementedError

    def to_file(self, filepath):
        raise NotImplementedError

    def _ms_to_c(self, matrix):
        if self._ndim == 2:
            assert (matrix[2] == [0., 0., 1.]).all()
            return matrix[:2].flatten()
        if self._ndim == 3:
            assert (matrix[3] == [0., 0., 0., 1]).all()
            return matrix[:3].flatten()

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
    am = AffineMatrix(parameters=[1.1, 0., 5, 0., 1.3, 2])
    print(am.get_matrix(order='C'))
    print(am.get_matrix(order='Cs'))
    print(am.get_matrix(order='M'))
    print(am.get_matrix(order='Ms'))
    print((-am).get_matrix(order='Ms'))
    print((am * am).get_matrix(order='Ms'))
    print((am * -am).get_matrix(order='Ms'))
    print((-am * am).get_matrix(order='Ms'))

