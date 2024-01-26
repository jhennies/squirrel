import SimpleITK as sitk
import numpy as np


def apply_transform(
        image,
        transform,
        verbose=False
):
    # TODO Figure out a way to do this with Transformix
    pass


def save_transforms(parameters, out_filepath, param_order='M', save_order='M', ndim=3, verbose=False):

    parameters = np.array(parameters)
    if verbose:
        print(f'parameters = {parameters}')

    def _elastix2m(param):
        pr = np.zeros(param.shape, dtype=param.dtype)
        pr[:ndim ** 2] = param[:ndim ** 2][::-1]
        pr[ndim ** 2:] = param[ndim ** 2:][::-1]
        param = pr

        pr = np.reshape(param[: ndim ** 2], (ndim, ndim), order='C')
        pr = np.concatenate([pr, np.array([param[ndim ** 2:]]).swapaxes(0, 1)], axis=1)
        return pr

        # print(f'param = {param}')
        # p = np.zeros(param.shape, dtype=param.dtype)
        # p[:ndim ** 2] = param[:ndim ** 2][::-1]
        # p[ndim ** 2:] = param[ndim ** 2:][::-1]
        # print(f'param = {p}')
        # return _f2m(p)

    def _c2m(param):
        return np.reshape(param, (ndim, ndim + 1), order='C')

    def _f2m(param):
        return np.reshape(param, (ndim, ndim + 1), order='F')

    def _m2c(param):
        return param.flatten(order='C')

    def _m2f(param):
        return param.flatten(order='F')

    def _change_order(params):
        if param_order == 'elastix':
            params = _elastix2m(params)
        if param_order == 'C':
            params = _c2m(params)
        if param_order == 'F':
            params = _f2m(params)
        if save_order == 'C':
            params = _m2c(params)
        if save_order == 'F':
            params = _m2f(params)
        return params

    if verbose:
        print(f'parameters.shape = {parameters.shape}')
        print(f'parameters.ndim = {parameters.ndim}')

    if param_order != save_order:
        if (param_order != 'M' and parameters.ndim == 2) or (param_order == 'M' and parameters.ndim == 3):
            parameters.tolist()
            for idx, p in enumerate(parameters):
                parameters[idx] = _change_order(p)
        else:
            parameters = _change_order(parameters)

    import json

    if verbose:
        print(f'parameters.shape = {parameters.shape}')

    if out_filepath is not None:
        with open(out_filepath, mode='w') as f:
            json.dump(parameters.tolist(), f, indent=2)

    return parameters


def register_with_elastix(
        fixed_image, moving_image,
        transform='affine',
        automatic_transform_initialization=False,
        out_dir=None,
        verbose=False
):

    if type(fixed_image) == np.ndarray:
        if verbose:
            print(f'Getting fixed image from array with shape = {fixed_image.shape}')
        fixed_image = sitk.GetImageFromArray(fixed_image)
    if type(moving_image) == np.ndarray:
        if verbose:
            print(f'Getting moving image from array with shape = {moving_image.shape}')
        moving_image = sitk.GetImageFromArray(moving_image)

    if verbose:
        print(f'fixed_image.GetSize() = {fixed_image.GetSize()}')
        print(f'moving_image.GetSize() = {moving_image.GetSize()}')

    # Set the input images
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)
    if out_dir is not None:
        elastixImageFilter.SetOutputDirectory(out_dir)
    elastixImageFilter.LogToConsoleOff()

    # Set the parameters
    parameter_map = sitk.GetDefaultParameterMap(transform)
    parameter_map['AutomaticTransformInitialization'] = ['true' if automatic_transform_initialization else 'false']
    elastixImageFilter.SetParameterMap(parameter_map)

    if verbose:
        print(f'Running Elastix with these parameters:')
        elastixImageFilter.PrintParameterMap()

    elastixImageFilter.Execute()
    result_image = elastixImageFilter.GetResultImage()
    result_transform_parameters = elastixImageFilter.GetTransformParameterMap()[0]['TransformParameters']

    return sitk.GetArrayFromImage(result_image), result_transform_parameters
