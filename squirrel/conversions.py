
def h5_to_nii():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert a dataset within a h5 container to a nifti volume',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('h5_file', type=str,
                        help='Input h5 container')
    parser.add_argument('h5_key', type=str,
                        help='Internal path of the dataset')
    parser.add_argument('out_filepath', type=str,
                        help='Output filepath where the results will be written to')
    parser.add_argument('-ax', '--axes_order', type=str, default='zyx',
                        help='Re-define the order of the volume axes')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    h5_file = args.h5_file
    h5_key = args.h5_key
    out_filepath = args.out_filepath
    axes_order = args.axes_order
    verbose = args.verbose

    from squirrel.workflows.convert import h5_to_nii_workflow

    h5_to_nii_workflow(
        h5_file,
        h5_key,
        out_filepath,
        axes_order=axes_order,
        verbose=verbose
    )


def h5_to_tif():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert a dataset within a h5 container to a tif stack',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('h5_file', type=str,
                        help='Input h5 container')
    parser.add_argument('h5_key', type=str,
                        help='Internal path of the dataset')
    parser.add_argument('out_folder', type=str,
                        help='Output folder where the results will be written to')
    parser.add_argument('-ax', '--axes_order', type=str, default='zyx',
                        help='Re-define the order of the volume axes')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    h5_file = args.h5_file
    h5_key = args.h5_key
    out_folder = args.out_folder
    axes_order = args.axes_order
    verbose = args.verbose

    from squirrel.workflows.convert import h5_to_tif_workflow

    h5_to_tif_workflow(
        h5_file,
        h5_key,
        out_folder,
        axes_order=axes_order,
        verbose=verbose
    )


def mib_to_tif():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert a Microscopy Image Browser model to a tif stack',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('mib_model_file', type=str,
                        help='Input MIB model file')
    parser.add_argument('out_folder', type=str,
                        help='Output folder where the results will be written to')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    mib_model_file = args.mib_model_file
    out_folder = args.out_folder
    verbose = args.verbose

    from squirrel.workflows.convert import mib_to_tif_workflow

    mib_to_tif_workflow(
        mib_model_file,
        out_folder,
        verbose=verbose
    )


def stack_to_ome_zarr():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert a dataset within a h5 container or a tif stack to ome.zarr',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('stack_path', type=str,
                        help='Input h5 container or tif stack')
    parser.add_argument('ome_zarr_filepath', type=str,
                        help='Output ome-zarr dataset')
    parser.add_argument('--stack_pattern', type=str, default='*.tif',
                        help='File pattern for globbing the input stack; default="*.tif"')
    parser.add_argument('--stack_key', type=str, default='data',
                        help='Path within input h5 file; default="data"')
    parser.add_argument('--resolution', type=float, nargs=3, default=(1., 1., 1.),
                        help='Resolution of input data; default=(1., 1., 1.)')
    parser.add_argument('--unit', type=str, default='pixel',
                        help='Unit of input resolution; default="pixel"')
    parser.add_argument('--downsample_type', type=str, default='Average',
                        help='Downsample type used to create the resolution pyramid; default="Average"')
    parser.add_argument('--downsample_factors', type=int, nargs='+', default=(2, 2, 2),
                        help='Downsample factors used to create the resolution pyramid; default=(2, 2, 2)')
    parser.add_argument('--name', type=str, default=None,
                        help='Name of the dataset; defaults to the filename (without ome.zarr extension)')
    parser.add_argument('--chunk_size', type=int, nargs=3, default=[1, 512, 512],
                        help='Chunk size of the ome-zarr dataset; default=[1, 512, 512]')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
    parser.add_argument('--save_bounds', action='store_true',
                        help='Saves a json file alongside that contains the bounds of the non-zero area of each slice')
    parser.add_argument('--append', action='store_true',
                        help='Set to true if the dataset already exists and the new data should be placed into it')
    parser.add_argument('--n_threads', type=int, default=1,
                        help='Number of CPUs to use')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    stack_path = args.stack_path
    ome_zarr_filepath = args.ome_zarr_filepath
    stack_pattern = args.stack_pattern
    stack_key = args.stack_key
    resolution = args.resolution
    unit = args.unit
    downsample_type = args.downsample_type
    downsample_factors = args.downsample_factors
    name = args.name
    chunk_size = args.chunk_size
    z_range = args.z_range
    save_bounds = args.save_bounds
    append = args.append
    n_threads = args.n_threads
    verbose = args.verbose

    from squirrel.workflows.convert import stack_to_ome_zarr_workflow

    stack_to_ome_zarr_workflow(
        stack_path,
        ome_zarr_filepath,
        stack_pattern=stack_pattern,
        stack_key=stack_key,
        resolution=resolution,
        unit=unit,
        downsample_type=downsample_type,
        downsample_factors=downsample_factors,
        name=name,
        chunk_size=chunk_size,
        z_range=z_range,
        save_bounds=save_bounds,
        append=append,
        n_threads=n_threads,
        verbose=verbose,
    )


def ome_zarr_to_stack():
    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert a ome.zarr dataset to a tif stack',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('ome_zarr_filepath', type=str,
                        help='Input ome-zarr dataset')
    parser.add_argument('target_dirpath', type=str,
                        help='Output tif stack')
    parser.add_argument('--ome_zarr_key', type=str, default='s0',
                        help='Path within input ome-zarr dataset; default="s0"')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
    parser.add_argument('--n_threads', type=int, default=1,
                        help='Number of CPUs to use')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    ome_zarr_filepath = args.ome_zarr_filepath
    target_dirpath = args.target_dirpath
    ome_zarr_key = args.ome_zarr_key
    z_range = args.z_range
    n_threads = args.n_threads
    verbose = args.verbose

    from squirrel.workflows.convert import ome_zarr_to_stack_workflow

    ome_zarr_to_stack_workflow(
        ome_zarr_filepath,
        target_dirpath,
        ome_zarr_key=ome_zarr_key,
        z_range=z_range,
        n_threads=n_threads,
        verbose=verbose,
    )


def n5_to_stack():
    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert a n5 dataset to a tif stack',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('n5_filepath', type=str,
                        help='Input n5 dataset')
    parser.add_argument('target_dirpath', type=str,
                        help='Output tif stack')
    parser.add_argument('--n5_key', type=str, default='setup0/timepoint0/s0',
                        help='Path within input n5 dataset; default="setup0/timepoint0/s0"')
    parser.add_argument('--z_range', type=int, nargs=2, default=None,
                        help='Use certain slices of the stack only; Defaults to the entire stack')
    parser.add_argument('--n_threads', type=int, default=1,
                        help='Number of CPUs to use')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    ome_zarr_filepath = args.ome_zarr_filepath
    target_dirpath = args.target_dirpath
    ome_zarr_key = args.ome_zarr_key
    z_range = args.z_range
    n_threads = args.n_threads
    verbose = args.verbose

    from squirrel.workflows.convert import n5_to_stack_workflow

    n5_to_stack_workflow(
        ome_zarr_filepath,
        target_dirpath,
        ome_zarr_key=ome_zarr_key,
        z_range=z_range,
        n_threads=n_threads,
        verbose=verbose,
    )


if __name__ == '__main__':
    stack_to_ome_zarr()
