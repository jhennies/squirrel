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
            'apply_affine_on_volume = squirrel.apply_transformation:affine_on_volume',
            'apply_rotation_and_scale = squirrel.apply_transformation:apply_rotation_and_scale',
            'average_affine_on_volume = squirrel.apply_transformation:average_affine_on_volume',
            'sequential_affine_on_volume = squirrel.apply_transformation:sequential_affine_on_volume',
            'compress_tif_stack = squirrel.compress_tif_stack:main',
            'elastix_affine3d = squirrel.elastix_registration:affine3d',
            'elastix_register_z_chunks = squirrel.elastix_registration:register_z_chunks',
            'h5_to_nii = squirrel.h5_to_nii:main',
            'h5_to_tif = squirrel.h5_to_tif:main',
            'mib_to_tif = squirrel.mib_to_tif:main',
            'normalize_slices = squirrel.normalize_slices:main',
            'sift3d = squirrel.sift3d:main',
            'tif_merge = squirrel.tif_merge:main',
            'view_in_napari = squirrel.view_in_napari:main'
        ]
    },
    install_requires=[
        'numpy',
        'h5py',
        'tifffile'
    ]
)
