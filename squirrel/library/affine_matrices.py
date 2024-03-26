
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

