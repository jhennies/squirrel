
from .io import make_directory, load_h5_container, write_tif_stack


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
    make_directory(out_folder)

    # Load data from the source file
    data = load_h5_container(h5_file, h5_key)

    # Write results
    write_tif_stack(data, out_folder)
