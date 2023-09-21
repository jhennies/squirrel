
import os


def _make_directory(directory, exist_ok=False, not_found_ok=False):
    try:
        os.mkdir(directory)
    except FileNotFoundError as e:
        if not_found_ok:
            return e
        raise
    except FileExistsError as e:
        if exist_ok:
            return e
        raise
    return None


def h5_to_tif(
        h5_file,
        h5_key,
        out_folder,
        verbose=False
):
    if verbose:
        print(f'h5_file = {h5_file}')
        print(f'h5_key = {h5_key}')
        print(f'out_folder = {out_folder}')
        print('test')

    # Create target folder (parent folder has to exist)
    # _make_directory(out_folder)

    # Open source

    # Write results
