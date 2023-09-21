import unittest
import warnings


class TestIO(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_make_directory(self):

        print('Testing make_directory ...')
        from squirrel.io import make_directory
        from random import randint
        from os.path import exists, join
        from os import rmdir

        test_dirname = f'test_io_{randint(1000, 9999)}'

        # Checking successful creation of the directory
        test_full_dir = f'./{test_dirname}'
        make_directory(test_full_dir)

        try:
            assert exists(test_full_dir)
            print('... directory created')

            # Check kwargs working
            assert make_directory(test_full_dir, exist_ok=True) == 'exists'
            print('... exist_ok')
            assert make_directory(join(test_full_dir, 'other', 'dir'), not_found_ok=True) == 'not_found'
            print('... path_not_found_ok')
            rmdir(test_full_dir)
        except Exception:
            rmdir(test_full_dir)
            raise

    def test_load_h5_container(self):

        print('Testing load_h5_container ...')
        from squirrel.io import load_h5_container
        from h5py import File
        from random import randint
        import numpy as np
        from os import remove

        test_array = np.random.rand(3, 4, 5)
        test_filepath = f'./test_load_h5_container_{randint(1000, 9999)}.h5'
        with File(test_filepath, mode='w') as f:
            f.create_dataset('data', data=test_array)

        # Testing the most important ones
        axes_orders = ['zyx', 'zxy', 'xyz']

        for axes_order in axes_orders:

            print(f'... testing for axes_order = {axes_order}')

            loaded_array = load_h5_container(test_filepath, 'data', axes_order=axes_order)
            try:
                if axes_order == 'zyx':
                    assert loaded_array.shape == test_array.shape
                if axes_order == 'zxy':
                    assert test_array.shape == (loaded_array.shape[0], loaded_array.shape[2], loaded_array.shape[1])
                if axes_order == 'xyz':
                    assert test_array.shape == (loaded_array.shape[2], loaded_array.shape[1], loaded_array.shape[0])
                print('...   correct shapes')
                assert loaded_array.sum() == test_array.sum()
                print('...   same sum')
            except Exception:
                remove(test_filepath)
                raise
        remove(test_filepath)

    def test_write_tif_stack(self):

        print('Testing write_tif_stack ...')
        from squirrel.io import write_tif_stack
        from random import randint
        import numpy as np
        from shutil import rmtree
        from os import listdir, mkdir

        test_folder = f'./test_io_{randint(1000, 9999)}'
        mkdir(test_folder)

        test_array = np.random.rand(5, 5, 5)

        try:
            write_tif_stack(test_array, test_folder)
            assert len(listdir(test_folder)) == 5
            print('... number of files matches')
            rmtree(test_folder)
        except Exception:
            rmtree(test_folder)
            raise
