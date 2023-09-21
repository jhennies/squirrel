from setuptools import setup

setup(
    name='squirrel',
    version='0.0.1',
    author='jhennies',
    author_email='hennies@embl.de',
    packages=['squirrel'],
    scripts=[
        'bin/h5_to_tif.py'
    ],
    install_requires=[
        'numpy',
        'h5py',
        'tifffile'
    ]
)
