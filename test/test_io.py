import unittest
import warnings
import os
from shutil import rmtree
import numpy as np
import tempfile

from random import randint


class TestIO(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_make_directory(self):

        print('Testing make_directory ...')
        from squirrel.library.io import make_directory

        test_dirname = f'test_io_{randint(1000, 9999)}'

        # Checking successful creation of the directory
        test_full_dir = f'./{test_dirname}'
        make_directory(test_full_dir)

        try:
            assert os.path.exists(test_full_dir)
            print('... directory created')

            # Check kwargs working
            assert make_directory(test_full_dir, exist_ok=True) == 'exists'
            print('... exist_ok')
            assert make_directory(os.path.join(test_full_dir, 'other', 'dir'), not_found_ok=True) == 'not_found'
            print('... path_not_found_ok')
            os.rmdir(test_full_dir)
        except Exception:
            os.rmdir(test_full_dir)
            raise

    def test_load_h5_container(self):

        print('Testing load_h5_container ...')
        from squirrel.library.io import load_h5_container
        from h5py import File

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
                os.remove(test_filepath)
                raise
        os.remove(test_filepath)

    def test_write_tif_stack(self):

        print('Testing write_tif_stack ...')
        from squirrel.library.io import write_tif_stack

        test_folder = f'./test_io_{randint(1000, 9999)}'
        os.mkdir(test_folder)

        test_array = np.random.rand(5, 5, 5)

        try:
            write_tif_stack(test_array, test_folder)
            assert len(os.listdir(test_folder)) == 5
            print('... number of files matches')
            rmtree(test_folder)
        except Exception:
            rmtree(test_folder)
            raise

    def test_get_file_list(self):

        print('Testing get_file_list ...')

        test_folder = f'./test_io_{randint(1000, 9999)}'

        os.mkdir(test_folder)

        open(os.path.join(test_folder, 'a.xyz'), mode='w').close()
        open(os.path.join(test_folder, 'b.xyz'), mode='w').close()
        open(os.path.join(test_folder, 'c.xyz'), mode='w').close()

        from squirrel.library.io import get_file_list
        file_list = get_file_list(test_folder, pattern='*.xyz')

        try:
            assert len(file_list) == 3
            print('... proper length')
            assert os.path.join(test_folder, 'a.xyz') in file_list, f"{os.path.join('.', test_folder, 'a.xyz')} not in {file_list}"
            assert os.path.join(test_folder, 'b.xyz') in file_list
            assert os.path.join(test_folder, 'c.xyz') in file_list
            print('... all files found!')
            rmtree(test_folder)
        except Exception:
            rmtree(test_folder)
            raise

    def test_read_tif_slice(self):

        print('Testing read_tif_slice ...')

        from tifffile import imwrite
        from squirrel.library.io import read_tif_slice

        arr = np.random.randint(0, 255, (10, 10), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            imwrite(f.name, arr)
            result, filename = read_tif_slice(f.name)

        np.testing.assert_array_equal(arr, result)

    def test_read_png_slice(self):

        print('Testing read_png_slice ...')

        from skimage import io as skio
        from squirrel.library.io import read_png_slice

        arr = np.random.randint(0, 255, (10, 10), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            skio.imsave(f.name, arr)
            result, filename = read_png_slice(f.name)

        np.testing.assert_array_equal(arr, result)

    def test_write_tif_slice(self):

        print('Testing write_tif_slice ...')

        test_file = f'test_io_{randint(1000, 9999)}.tif'

        from tifffile import imread
        from squirrel.library.io import write_tif_slice
        test_array = np.random.rand(3, 4)

        try:
            write_tif_slice(test_array, './', test_file)
            loaded_array = imread(test_file)
            assert (loaded_array == test_array).all()
            print('... Data is identical')
            os.remove(test_file)
        except Exception:
            os.remove(test_file)
            raise


class TestGenericStack(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_get_item(self):

        print('Testing _GenericStack.test_get_item ...')

        from pathlib import Path
        from skimage import io as skio
        from squirrel.library.io import _GenericStack

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # create three PNG images
            arrs = []
            for i in range(3):
                arrs.append(np.full((10, 10), i, dtype=np.uint8))
                skio.imsave(tmp_path / f"slice_{i}.png", arrs[-1])
            arrs = np.array(arrs)

            stack = _GenericStack(tmpdir, image_type='png', pattern='*.png')

            np.testing.assert_array_equal(stack, arrs)
            np.testing.assert_array_equal(stack[1:2], arrs[1:2])

            assert stack.shape == stack.get_shape(check_all=True) == [3, 10, 10]

    def test_get_item_inconsistent_shapes(self):

        print('Testing _GenericStack.test_get_item with inconsistent shapes ...')

        from pathlib import Path
        from skimage import io as skio
        from squirrel.library.io import _GenericStack

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # create three PNG images
            shapes = [[7, 10], [8, 9], [9, 7]]
            arrs = []
            for i in range(3):
                arrs.append(np.full(shapes[i], i, dtype=np.uint8))
                skio.imsave(tmp_path / f"slice_{i}.png", arrs[-1])

            stack = _GenericStack(tmpdir, image_type='png', pattern='*.png')

            for i in range(3):
                np.testing.assert_array_equal(stack[i], arrs[i])
            np.testing.assert_array_equal(stack[1:2], arrs[1][None, :])

            assert stack.shape == [3, 7, 10]
            assert stack.get_shape(check_all=True) == [3, 9, 10]


