# squirrel

Python-based library for conversion and processing of 3D EM data.

## Installation via conda

hopefully coming soon:

```
conda create -n squirrel-env squirrel
```

## Installation via pip

```
mamba create -n squirrel-env -c conda-forge python=3.11
mkdir path/to/src
cd path/to/src
git clone https://github.com/jhennies/squirrel
pip install -e squirrel
```

## Installation of optional packages

Install the following as necessary to enable additional functionality

### nibabel for handling nii files
This is required for 3D SIFT
```
mamba install -c conda-forge nibabel
```

### Napari for 3D visualization
```
mamba install -c conda-forge napari pyqt
```

### For registration
```
pip install SimpleITK-SimpleElastix
pip install transforms3d
```
also install OpenCV (see below)

### OpenCV for 2D SIFT 
```
mamba install -c conda-forge opencv
```

### For ome.zarr support
```
mamba install -c conda-forge zarr
```

## Usage

The main functions can be used directly after activation of the 
conda environment:

```
conda activate squirrel-env

compress_tif_stack -h
h5_to_tif -h
mib_to_tif -h
normalize_slices -h
tif_merge -h
```

The -h flag yields the help output describing functions and parameters
