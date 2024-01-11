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
            'compress_tif_stack = squirrel.scripts.compress_tif_stack:main',
            'h5_to_tif = squirrel.scripts.h5_to_tif:main',
            'mib_to_tif = squirrel.scripts.mib_to_tif:main',
            'normalize_slices = squirrel.scripts.normalize_slices:main',
            'tif_merge = squirrel.scripts.tif_merge:main'
        ]
    },
    install_requires=[
        'numpy',
        'h5py',
        'tifffile'
    ]
)
