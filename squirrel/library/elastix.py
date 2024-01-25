import SimpleITK as sitk
import numpy as np


def apply_transform(
        image,
        transform,
        verbose=False
):
    # TODO Figure out a way to do this with Transformix
    pass

#     if verbose:
#         print(f'transform = {transform}')
#
#     import SimpleITK as sitk
#
#     if type(image) == np.ndarray:
#         if verbose:
#             print(f'Getting fixed image from array with shape = {image.shape}')
#         image = sitk.GetImageFromArray(image)
#
#     pmap = sitk.GetDefaultParameterMap('affine')
#
#     result = sitk.Transformix(image, pmap)
#
#     # pmap = sitk.AffineTransform()
#     # pmap['TransformParameters'] = [str(x) for x in transform]
#     # print(pmap)
#     # print(pmap['TransformParameters'])
#     # result = sitk.Transformix(image, pmap)
#
#     # tx = sitk.TransformixImageFilter()
#     # tx.SetTransformParameterMap(sitk.GetDefaultParameterMap('affine'))
#     # tx.SetTransformParameter('TransformParameters', [str(x) for x in transform])
#     # tx.SetMovingImage(image)
#     # print(tx.GetTransformParameterMap()[0]['TransformParameters'])
#     # tx.Execute()
#     # result = tx.GetResultImage()
#
#     from squirrel.library.io import write_h5_container
#     write_h5_container(
#         '/media/julian/Data/projects/em-xray-alignment/automated_registration_sift3d/step02_registration_refinement/b-avg_transform_on_xray/tmp.h5',
#         result
#     )
#
#
# from squirrel.library.io import load_data
#
# apply_transform(
#     load_data('/media/julian/Data/projects/em-xray-alignment/automated_registration_sift3d/step01_registration_xray_to_em_sift3d/warped.nii'),
#     [
#         1.00427,
#         0.00617366,
#         0.0186604,
#         -0.00520014,
#         0.985899,
#         0.00638913,
#         -0.0174637,
#         -0.0295123,
#         0.921393,
#         0.720291,
#         -1.27948,
#         -0.0952311
#     ],
#     verbose=True
# )


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
