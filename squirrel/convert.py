
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
