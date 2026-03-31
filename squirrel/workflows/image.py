
import os


def filter_2d_workflow(
        input_filepath,
        out_filepath,
        filters=None,
        verbose=False
):
    if verbose:
        print(f'input_filepath = {input_filepath}')
        print(f'out_filepath = {out_filepath}')
        print(f'filters = {filters}')

    from squirrel.library.filters import ImageFilter

    from squirrel.library.io import read_tif_slice, write_tif_slice
    img = read_tif_slice(input_filepath, return_filepath=False)

    imf = ImageFilter(img)
    filtered_image = imf.get_filtered(filters)

    write_tif_slice(filtered_image, *os.path.split(out_filepath))
