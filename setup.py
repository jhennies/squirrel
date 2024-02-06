import runpy
from setuptools import setup

version = runpy.run_path("squirrel/__version__.py")["__version__"]

setup(
    name='squirrel',
    version=version,
    author='Julian Hennies',
    author_email='hennies@embl.de',
    url='https://github.com/jhennies/squirrel',
    license="GPLv3",
    packages=['squirrel'],
    entry_points={
        'console_scripts': [
            'apply_stack_alignment = squirrel.apply_transformation:apply_stack_alignment',
            'apply_affine_on_volume = squirrel.apply_transformation:affine_on_volume',
            'apply_rotation_and_scale = squirrel.apply_transformation:apply_rotation_and_scale',
            'apply_z_chunks_to_volume = squirrel.apply_transformation:apply_z_chunks_to_volume',
            'average_affine_on_volume = squirrel.apply_transformation:average_affine_on_volume',
            'sequential_affine_on_volume = squirrel.apply_transformation:sequential_affine_on_volume',
            'decompose_affine_matrix = squirrel.apply_transformation:decompose_affine_matrix',
            'compress_tif_stack = squirrel.compress_tif_stack:main',
            'elastix_on_volume3d = squirrel.elastix_registration:elastix_on_volume3d',
            'elastix_slices_to_volume = squirrel.elastix_registration:elastix_slices_to_volume',
            'elastix_register_z_chunks = squirrel.elastix_registration:register_z_chunks',
            'elastix_stack_alignment = squirrel.elastix_registration:elastix_stack_alignment',
            'h5_to_nii = squirrel.h5_to_nii:main',
            'h5_to_tif = squirrel.h5_to_tif:main',
            'mib_to_tif = squirrel.mib_to_tif:main',
            'normalize_slices = squirrel.normalize_slices:main',
            'sift2d_stack_alignment = squirrel.sift2d:sift2d_stack_alignment',
            'sift3d = squirrel.sift3d:main',
            'tif_merge = squirrel.tif_merge:main',
            'view_in_napari = squirrel.view_in_napari:main',
            'linalg_dot_product_on_affines = squirrel.linear_algebra:dot_product_on_affines',
            'linalg_scale_sequential_affines = squirrel.linear_algebra:scale_sequential_affines',
            'linalg_apply_affine_sequence = squirrel.linear_algebra:apply_affine_sequence',
            'linalg_smooth_affine_sequence = squirrel.linear_algebra:smooth_affine_sequence'
        ]
    },
    install_requires=[
        'numpy',
        'h5py',
        'tifffile'
    ]
)
