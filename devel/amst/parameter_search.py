
from squirrel.workflows.amst import amst_workflow
from squirrel.workflows.transformation import apply_stack_alignment_on_volume_workflow
import os
import numpy as np


def do_parameter_search(
        out_dirpath,
        parameter_dict,
        parameter_groups,
        current_parameters,
        pre_aligned_stack,
        rois,
        max_iterations=16,
):

    def to_string(p):
        return [str(p)]

    def current_params_from_dict(pdict, pgroups, curpars, group, as_str=False):
        if as_str:
            return {k: to_string(pdict[k][curpars[k]]) for k in pgroups[group]}
        return {k: pdict[k][curpars[k]] for k in pgroups[group]}

    def get_elastix_parameter_map(pdict, pgroups, curpars):

        from SimpleITK import ParameterMap
        ParameterMap()
        from SimpleITK import GetDefaultParameterMap

        # parameter_map = GetDefaultParameterMap(
        #     'affine', numberOfResolutions=1, finalGridSpacingInPhysicalUnits=8.0
        # )
        parameter_map = GetDefaultParameterMap(
            'affine', **current_params_from_dict(pdict, pgroups, curpars, 'elastix_init', as_str=False)
        )

        parameter_map['AutomaticParameterEstimation'] = ('true',)
        parameter_map['Interpolator'] = ('BSplineInterpolator',)
        parameter_map['FixedImagePyramid'] = ('FixedRecursiveImagePyramid',)
        parameter_map['MovingImagePyramid'] = ('MovingRecursiveImagePyramid',)
        parameter_map['AutomaticScalesEstimation'] = ('true',)
        parameter_map['MaximumNumberOfIterations'] = ('256',)
        parameter_map['ImageSampler'] = ('RandomCoordinate',)
        parameter_map['ErodeMask'] = ('true',)
        parameter_map['NumberOfSpatialSamples'] = ('2048',)
        parameter_map['NumberOfHistogramBins'] = ('64',)
        parameter_map['BSplineInterpolationOrder'] = ('3',)
        parameter_map['NumberOfSamplesForExactGradient'] = ('4096',)

        for k, v in current_params_from_dict(pdict, pgroups, curpars, 'elastix', as_str=True).items():
            parameter_map[k] = v

        return parameter_map

    from squirrel.library.io import load_data_handle, write_h5_container
    from squirrel.workflows.elastix import elastix_stack_alignment_workflow
    from squirrel.library.affine_matrices import AffineStack

    def get_amst_score(data_filepaths, transform_filepaths):

        translations = []
        for idx in range(len(data_filepaths)):
            data_filepath = data_filepaths[idx]
            transform_filepath = transform_filepaths[idx]
            elastix_stack_alignment_workflow(
                data_filepath,
                transform_filepath,
                number_of_spatial_samples=128,
                number_of_resolutions=1,
                transform='translation',
                quiet=True,
                overwrite=True
            )
            transforms = AffineStack(filepath=transform_filepath)
            translations.extend(transforms.get_translations()[1:])
        translations = np.array(translations)
        sq_deltas = translations[:, 0] ** 2 + translations[:, 1] ** 2
        return np.prod(sq_deltas) ** (1 / len(translations))

    def amst_wf_and_score(curpars, out_name, from_pre_align=False, remove_h5=False):

        transform_filepath = os.path.join(out_dirpath, f'{out_name}.json')
        aligned_stack_filepath = os.path.join(out_dirpath, f'{out_name}.h5')
        score_data_filepath = os.path.join(out_dirpath, f'{out_name}-score_data' + '{:02d}.h5')
        score_transforms_filepath = os.path.join(out_dirpath, f'{out_name}-score_transforms' + '{:02d}.json')

        param_map = get_elastix_parameter_map(parameter_dict, parameter_groups, curpars)
        # print('Calling AMST with:')
        # print(param_map)
        # print(param_map.items())
        # print(f'and median_radius = {parameter_dict["median_radius"][curpars["median_radius"]]}')

        if not from_pre_align:
            amst_workflow(
                pre_aligned_stack,
                transform_filepath,
                pre_align_key='s0',
                transform='affine',
                median_radius=parameter_dict['median_radius'][curpars['median_radius']],
                elastix_parameters=param_map,
                quiet=True
                # z_range=[8, 16]
            )
            apply_stack_alignment_on_volume_workflow(
                pre_aligned_stack,
                transform_filepath,
                aligned_stack_filepath,
                key='s0',
                auto_pad=False,
                # z_range=[8, 16],
                quiet=True,
                n_workers=16
            )
            h, shp = load_data_handle(aligned_stack_filepath, key='data')
        else:
            h, shp = load_data_handle(pre_aligned_stack, key='s0')
        score_data_filepaths = []
        score_transforms_filepaths = []
        for roi_idx, roi in enumerate(rois):
            score_data_filepaths.append(score_data_filepath.format(roi_idx))
            score_transforms_filepaths.append(score_transforms_filepath.format(roi_idx))
            score_data = h[roi]
            write_h5_container(score_data_filepaths[-1], score_data)
        if remove_h5:
            os.remove(aligned_stack_filepath)
        return get_amst_score(score_data_filepaths, score_transforms_filepaths)

    def update_current_parameters(curpars, par_scores, ref, update_best=1):
        print('>>>> Updating current parameters')
        print(curpars)

        change_detected = False

        scores = []
        param_names = []
        param_vals = []
        for this_param, this_scores in par_scores.items():
            this_min_score = 1000
            for this_value, score in this_scores.items():
                if score < ref and score < this_min_score:
                    scores.append(score)
                    param_names.append(this_param)
                    param_vals.append(this_value)

        order = np.argsort(scores)
        sorted_scores = [scores[x] for x in order]
        sorted_param_names = [param_names[x] for x in order]
        sorted_param_vals = [param_vals[x] for x in order]

        if len(sorted_scores) > 0:
            change_detected = True
            for idx in range(update_best):
                if len(sorted_scores) > idx:
                    curpars[sorted_param_names[idx]] = sorted_param_vals[idx]

        print(curpars)
        print('<<<<')

        return curpars, change_detected

    pre_align_score = amst_wf_and_score(current_parameters, 'pre-align-ref', from_pre_align=True)
    print('\n------------------------------------------------------')
    print(f'pre_align_score = {pre_align_score}')
    print('------------------------------------------------------\n')

    change = True
    all_ref_scores = []
    idx = 0
    while change and idx < max_iterations:
        print(f'idx = {idx}')

        ref_score = amst_wf_and_score(current_parameters, 'amst-{:04d}-ref'.format(idx))
        print('\n------------------------------------------------------')
        print(f'Starting iteration {idx} with ref_score = {ref_score}')
        if all_ref_scores:
            print(f'Previous ref_score = {all_ref_scores[-1]}')
        print('------------------------------------------------------\n')
        all_ref_scores.append(ref_score)

        scores = dict()

        for this_param in current_parameters.keys():

            to_compute = []
            if current_parameters[this_param] > 0:
                to_compute.append(current_parameters[this_param] - 1)
            if current_parameters[this_param] > 1:
                to_compute.append(current_parameters[this_param] - 2)
            if current_parameters[this_param] < len(parameter_dict[this_param]) - 1:
                to_compute.append(current_parameters[this_param] + 1)
            if current_parameters[this_param] < len(parameter_dict[this_param]) - 2:
                to_compute.append(current_parameters[this_param] + 2)

            this_param_scores = dict()
            for tc in to_compute:
                cp_in = current_parameters.copy()
                cp_in[this_param] = tc
                this_param_scores[tc] = amst_wf_and_score(
                    cp_in, 'amst-{:04d}-{}-{}'.format(idx, this_param, tc), remove_h5=True
                )
                print(f'param: {this_param}, value: {tc}, score: {this_param_scores[tc]}')
            scores[this_param] = this_param_scores
            print('..................')
            print(f'scores = {scores}')
            print('..................')

        current_parameters, change = update_current_parameters(current_parameters, scores, ref_score)

        idx += 1

    final_score = amst_wf_and_score(current_parameters, 'amst-{:04d}-final'.format(idx))
    print('\n------------------------------------------------------')
    print(f'final_score = {final_score}')
    print(f'Previous ref_score = {all_ref_scores[-1]}')
    print(f'Initial ref_score = {all_ref_scores[0]}')
    print(f'Min ref_score = {np.min(all_ref_scores)} in iteration {np.argmin(all_ref_scores)}')
    print(f'pre_align_score = {pre_align_score}')
    print('------------------------------------------------------\n')


def search_00():
    pre_aligned_stack = '/media/julian/Data/projects/kors/align/4T/amst_parameter_test/pre_align/pre-align.ome.zarr'
    out_dirpath = '/media/julian/Data/projects/kors/align/4T/amst_parameter_test/amst_results/'
    if not os.path.exists(out_dirpath):
        os.mkdir(out_dirpath)

    parameter_dict = dict(
        median_radius=[3, 4, 5, 6, 7, 8, 9],
        numberOfResolutions=[1, 2, 3, 4, 5, 6],
        finalGridSpacingInPhysicalUnits=[8.],
        FixedImagePyramid=['FixedRecursiveImagePyramid', 'FixedSmoothingImagePyramid'],
        MovingImagePyramid=['MovingRecursiveImagePyramid', 'MovingSmoothingImagePyramid'],
        AutomaticScalesEstimation=['true', 'false'],
        MaximumNumberOfIterations=[64, 128, 256, 512, 1024, 2048],
        NumberOfSpatialSamples=[256, 512, 1024, 2048, 4096, 8192],
        NumberOfHistogramBins=[16, 32, 64, 128],
        NumberOfSamplesForExactGradient=[512, 1024, 2048, 4096, 8192, 16384]
    )

    # current_parameters = dict(
    #     median_radius=0,
    #     numberOfResolutions=0,
    #     finalGridSpacingInPhysicalUnits=0,
    #     FixedImagePyramid=0,
    #     MovingImagePyramid=0,
    #     AutomaticScalesEstimation=0,
    #     MaximumNumberOfIterations=0,
    #     NumberOfSpatialSamples=0,
    #     NumberOfHistogramBins=0,
    #     NumberOfSamplesForExactGradient=0
    # )

    # 01
    current_parameters = dict(
        median_radius=0,
        numberOfResolutions=0,
        finalGridSpacingInPhysicalUnits=0,
        FixedImagePyramid=0,
        MovingImagePyramid=0,
        AutomaticScalesEstimation=0,
        MaximumNumberOfIterations=2,
        NumberOfSpatialSamples=2,
        NumberOfHistogramBins=2,
        NumberOfSamplesForExactGradient=0
    )

    # 02
    current_parameters = dict(
        median_radius=1,
        numberOfResolutions=1,
        finalGridSpacingInPhysicalUnits=0,
        FixedImagePyramid=0,
        MovingImagePyramid=0,
        AutomaticScalesEstimation=0,
        MaximumNumberOfIterations=4,
        NumberOfSpatialSamples=2,
        NumberOfHistogramBins=2,
        NumberOfSamplesForExactGradient=1
    )

    parameter_groups = dict(
        amst=['median_radius'],
        elastix_init=['numberOfResolutions', 'finalGridSpacingInPhysicalUnits'],
        elastix=[
            'FixedImagePyramid',
            'MovingImagePyramid',
            'AutomaticScalesEstimation',
            'MaximumNumberOfIterations',
            'NumberOfSpatialSamples',
            'NumberOfHistogramBins',
            'NumberOfSamplesForExactGradient'
        ]
    )

    max_iterations = 16

    roi = np.s_[
        :, 330: 458, 2518: 2646
    ]

    do_parameter_search(
        out_dirpath,
        parameter_dict,
        parameter_groups,
        current_parameters,
        pre_aligned_stack,
        roi,
        max_iterations=max_iterations
    )


def search_01():
    # Note: refining search_00

    pre_aligned_stack = '/media/julian/Data/projects/kors/align/4T/amst_parameter_test/pre_align/pre-align.ome.zarr'
    out_dirpath = '/media/julian/Data/projects/kors/align/4T/amst_parameter_test/amst_results_01/'
    if not os.path.exists(out_dirpath):
        os.mkdir(out_dirpath)

    parameter_dict = dict(
        median_radius=[3, 4, 5, 6, 7, 8, 9],
        numberOfResolutions=[1, 2, 3, 4, 5, 6],
        finalGridSpacingInPhysicalUnits=[8.],
        FixedImagePyramid=['FixedRecursiveImagePyramid'],
        MovingImagePyramid=['MovingRecursiveImagePyramid'],
        AutomaticScalesEstimation=['true'],
        MaximumNumberOfIterations=[512, 768, 1024, 1270, 1536, 2048],
        NumberOfSpatialSamples=[512, 768, 1024, 1270, 1536, 2048],
        NumberOfHistogramBins=[32, 48, 64, 80, 96, 128],
        NumberOfSamplesForExactGradient=[512, 768, 1024, 1270, 1536, 2048]
    )

    # start
    current_parameters = dict(
        median_radius=1,
        numberOfResolutions=1,
        finalGridSpacingInPhysicalUnits=0,
        FixedImagePyramid=0,
        MovingImagePyramid=0,
        AutomaticScalesEstimation=0,
        MaximumNumberOfIterations=2,
        NumberOfSpatialSamples=2,
        NumberOfHistogramBins=2,
        NumberOfSamplesForExactGradient=2
    )
    # it1
    current_parameters = dict(
        median_radius=1,
        numberOfResolutions=1,
        finalGridSpacingInPhysicalUnits=0,
        FixedImagePyramid=0,
        MovingImagePyramid=0,
        AutomaticScalesEstimation=0,
        MaximumNumberOfIterations=2,
        NumberOfSpatialSamples=2,
        NumberOfHistogramBins=1,
        NumberOfSamplesForExactGradient=2
    )

    parameter_groups = dict(
        amst=['median_radius'],
        elastix_init=['numberOfResolutions', 'finalGridSpacingInPhysicalUnits'],
        elastix=[
            'FixedImagePyramid',
            'MovingImagePyramid',
            'AutomaticScalesEstimation',
            'MaximumNumberOfIterations',
            'NumberOfSpatialSamples',
            'NumberOfHistogramBins',
            'NumberOfSamplesForExactGradient'
        ]
    )

    max_iterations = 16

    roi = np.s_[
          :, 330: 458, 2518: 2646
          ]

    do_parameter_search(
        out_dirpath,
        parameter_dict,
        parameter_groups,
        current_parameters,
        pre_aligned_stack,
        roi,
        max_iterations=max_iterations
    )


def search_02():
    # Multiple roi

    pre_aligned_stack = '/media/julian/Data/projects/kors/align/4T/amst_parameter_test/pre_align/pre-align.ome.zarr'
    out_dirpath = '/media/julian/Data/projects/kors/align/4T/amst_parameter_test/amst_results_02/'
    if not os.path.exists(out_dirpath):
        os.mkdir(out_dirpath)

    parameter_dict = dict(
        median_radius=[3, 4, 5, 6, 7, 8, 9],
        numberOfResolutions=[1, 2, 3, 4, 5, 6],
        finalGridSpacingInPhysicalUnits=[8.],
        FixedImagePyramid=['FixedRecursiveImagePyramid'],
        MovingImagePyramid=['MovingRecursiveImagePyramid'],
        AutomaticScalesEstimation=['true'],
        MaximumNumberOfIterations=[512, 768, 1024, 1270, 1536, 2048],
        NumberOfSpatialSamples=[512, 768, 1024, 1270, 1536, 2048],
        NumberOfHistogramBins=[32, 48, 64, 80, 96, 128],
        NumberOfSamplesForExactGradient=[512, 768, 1024, 1270, 1536, 2048]
    )

    # start
    current_parameters = dict(
        median_radius=1,
        numberOfResolutions=1,
        finalGridSpacingInPhysicalUnits=0,
        FixedImagePyramid=0,
        MovingImagePyramid=0,
        AutomaticScalesEstimation=0,
        MaximumNumberOfIterations=2,
        NumberOfSpatialSamples=2,
        NumberOfHistogramBins=1,
        NumberOfSamplesForExactGradient=2
    )

    parameter_groups = dict(
        amst=['median_radius'],
        elastix_init=['numberOfResolutions', 'finalGridSpacingInPhysicalUnits'],
        elastix=[
            'FixedImagePyramid',
            'MovingImagePyramid',
            'AutomaticScalesEstimation',
            'MaximumNumberOfIterations',
            'NumberOfSpatialSamples',
            'NumberOfHistogramBins',
            'NumberOfSamplesForExactGradient'
        ]
    )

    max_iterations = 16

    roi = [
        np.s_[:, 330: 458, 2518: 2646],  # Right edge
        np.s_[:, 386: 514, 440: 568],  # Left edge
        np.s_[:, 650: 778, 1640: 1768]  # Bottom
    ]

    do_parameter_search(
        out_dirpath,
        parameter_dict,
        parameter_groups,
        current_parameters,
        pre_aligned_stack,
        roi,
        max_iterations=max_iterations
    )


if __name__ == '__main__':

    search_02()
