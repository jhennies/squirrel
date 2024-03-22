
import os
import numpy as np


def template_matching_stack_alignment_workflow(
        stack,
        out_filepath,
        template_roi,
        search_roi=None,
        key='data',
        pattern='*.tif',
        z_range=None,
        save_template=False,
        verbose=False
):

    if os.path.exists(out_filepath):
        print(f'Target file exists: {out_filepath}\nSkipping elastix stack alignment workflow ...')
        return

    from ..library.io import load_data_handle, load_data_from_handle_stack, crop_roi
    from ..library.elastix import save_transforms
    from ..library.template_matching import match_template_on_image
    from ..library.transformation import save_transformation_matrices

    stack_h, stack_size = load_data_handle(stack, key=key, pattern=pattern)

    template = crop_roi(stack_h, template_roi)

    transforms = []

    if z_range is None:
        z_range = [0, stack_size[0]]

    for idx in range(*z_range):

        print(f'idx = {idx} / {z_range[1]}')

        if search_roi is None:
            z_slice, _ = load_data_from_handle_stack(stack_h, idx)
        else:
            z_slice, _ = crop_roi(stack_h, search_roi + [idx])

        transform = match_template_on_image(
            z_slice,
            template
        )
        print(transform)

        transforms.append(transform)

    transforms = np.array(transforms).astype(int).tolist()

    save_transformation_matrices(out_filepath, transforms, sequenced=True)

    if save_template:
        template_filepath = os.path.splitext(out_filepath)[0] + '.template.tif'
        from tifffile import imwrite
        imwrite(template_filepath, template)
