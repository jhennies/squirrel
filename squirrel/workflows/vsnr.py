
def vsnr_on_image(
        in_filepath,
        out_filepath=None,
        vsnr_param_filepath=None,
        verbose=False
):

    if verbose:
        print(f'in_filepath = {in_filepath}')
        print(f'out_filepath = {out_filepath}')

    from tifffile import imread, imwrite

    vsnr_params = get_vsnr_params(vsnr_param_filepath)

    corrected = vsnr_workflow(
        imread(in_filepath),
        vsnr_params=vsnr_params,
        verbose=verbose
    )
    imwrite(out_filepath, corrected, compression='zlib')


def vsnr_on_stack(
        in_folder,
        out_folder,
        in_pattern='*.tif',
        vsnr_param_filepath=None,
        overwrite=False,
        verbose=False
):

    if verbose:
        print(f'in_folder = {in_folder}')
        print(f'out_folder = {out_folder}')

    from glob import glob
    import os

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    im_list = sorted(glob(os.path.join(in_folder, in_pattern)))
    out_filepaths = [os.path.join(out_folder, os.path.split(fp)[1]) for fp in im_list]

    for idx, im_filepath in enumerate(im_list):

        if not overwrite and os.path.exists(out_filepaths[idx]):
            continue

        vsnr_on_image(
            im_filepath, out_filepath=out_filepaths[idx], vsnr_param_filepath=vsnr_param_filepath, verbose=verbose
        )
