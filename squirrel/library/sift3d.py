
import subprocess
import os


def run_sift3d(
        fixed_vol, moving_vol, out_path, omit_run=False,
        nn_thresh=0.8,
        err_thresh=5,
        peak_thresh=0.1,
        corner_thresh=0.4,
        num_kp_levels=3,
        num_iter=500
):

    matches_fp = os.path.join(out_path, 'matches.csv')
    transform_fp = os.path.join(out_path, 'affine.csv')
    warped_fp = os.path.join(out_path, 'warped.nii')
    concat_fp = os.path.join(out_path, 'concat.nii')
    keys_fp = os.path.join(out_path, 'keys.nii')
    lines_fp = os.path.join(out_path, 'lines.nii')

    if not omit_run:
        run_sift_command = [
            '/home/julian/src/SIFT3D/build/bin/regSift3D',
            '--matches', matches_fp,
            '--transform', transform_fp,
            '--warped', warped_fp,
            '--concat', concat_fp,
            '--keys', keys_fp,
            '--lines', lines_fp,
            '--nn_thresh', str(nn_thresh),
            '--err_thresh', str(err_thresh),
            '--peak_thresh', str(peak_thresh),
            '--corner_thresh', str(corner_thresh),
            '--num_kp_levels', str(num_kp_levels),
            '--num_iter', str(num_iter),
            moving_vol,
            fixed_vol
        ]
        subprocess.run(run_sift_command)

    return matches_fp, transform_fp, warped_fp, concat_fp, keys_fp, lines_fp
