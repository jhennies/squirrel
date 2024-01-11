import os.path


def sift3d_workflow(
        moving_filepath,
        fixed_filepath,
        out_path,
        moving_key='data',
        fixed_key='data',
        nn_thresh=0.8,
        corner_thresh=0.4,
        num_kp_levels=3,
        view_results_in_napari=False,
        force_run=False,
        verbose=False
):

    if verbose:
        print(f'moving_filepath = {moving_filepath}')
        print(f'fixed_filepath = {fixed_filepath}')
        print(f'out_path = {out_path}')
        print(f'moving_key = {moving_key}')
        print(f'fixed_key = {fixed_key}')
        print(f'nn_thresh = {nn_thresh}')
        print(f'corner_thresh = {corner_thresh}')
        print(f'num_kp_levels = {num_kp_levels}')

    from squirrel.library.io import get_filetype, load_nii_file
    from squirrel.library.sift3d import run_sift3d
    from squirrel.workflows.convert import h5_to_nii

    # FIXME This should be moved to the library
    def _create_nii_file(h5_filepath, h5_key, out_path):
        nii_filepath = os.path.join(
            out_path, os.path.splitext(os.path.split(h5_filepath)[1])[0] + '.nii'
        )
        h5_to_nii(h5_filepath, h5_key, nii_filepath)
        return nii_filepath

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    else:
        assert force_run or not os.listdir(out_path), 'The target directory is not empty! Use an empty or non-existing directory.'

    if get_filetype(moving_filepath) == 'h5':
        moving_filepath = _create_nii_file(moving_filepath, moving_key, out_path)
    if get_filetype(fixed_filepath) == 'h5':
        fixed_filepath = _create_nii_file(fixed_filepath, fixed_key, out_path)

    matches_fp, transform_fp, warped_fp, concat_fp, keys_fp, lines_fp = run_sift3d(
        fixed_filepath, moving_filepath, out_path, omit_run=False,
        nn_thresh=nn_thresh,
        corner_thresh=corner_thresh,
        num_kp_levels=num_kp_levels
    )

    if view_results_in_napari:
        from squirrel.workflows.viewing import view_in_napari

        view_in_napari(
            images=[moving_filepath, fixed_filepath, warped_fp, concat_fp],
            image_names=['Moving', 'Fixed', 'Warped', 'Concatenated'],
            labels=[keys_fp, lines_fp],
            label_names=['Keys', 'Lines'],
            verbose=verbose
        )
