

def sift_stack_alignment(
        stack,
        out_filepath,
        transform='translation',
        key='data',
        pattern='*.tif',
        verbose=False
):

    from ..library.io import load_data_handle, load_data_from_handle_stack
    from ..library.sift2d import register_with_sift
    from ..library.elastix import save_transforms

    stack, stack_size = load_data_handle(stack, key=key, pattern=pattern)

    transforms = []

    for idx in range(1, stack_size):

        z_slice_fixed = load_data_from_handle_stack(stack, idx - 1)
        z_slice_moving = load_data_from_handle_stack(stack, idx)

        transform_params = register_with_sift(
            z_slice_fixed,
            z_slice_moving,
            transform=transform,
            verbose=verbose
        )

        transforms.append(
            save_transforms(
                transform_params, None,
                param_order='M',  # TODO what does register_with_sift return?
                save_order='C',
                ndim=2,
                verbose=verbose
            ).tolist()
        )

    import json
    with open(out_filepath, mode='w') as f:
        json.dump(transforms, f, indent=2)
