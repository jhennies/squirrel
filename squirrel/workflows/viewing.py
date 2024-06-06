
import numpy as np


def _get_data(in_data, key, invert=False):

    if type(in_data) is not np.ndarray:
        # from squirrel.library.io import load_data
        from squirrel.library.io import load_data_handle
        return load_data_handle(in_data, key)[0][:]
        # return load_data(in_data, key=key, invert=invert)

    if invert:
        from ..library.data import invert_data
        return invert_data(in_data)

    return in_data


def view_in_napari(
        images=None,
        labels=None,
        image_keys=None,
        label_keys=None,
        image_names=None,
        label_names=None,
        invert_images=False,
        verbose=False
):
    import napari

    if verbose:
        print(f'images = {images}')
        print(f'labels = {labels}')
        print(f'image_keys = {image_keys}')
        print(f'label_keys = {label_keys}')
        print(f'image_names = {image_names}')
        print(f'label_names = {label_names}')

    from squirrel.library.viewing import add_labels_to_napari, add_images_to_napari

    viewer = napari.Viewer()

    if images is not None:
        if image_keys is None:
            image_keys = ['data'] * len(images)
        images = [
            _get_data(image_fp, image_keys[idx], invert=invert_images)
            for idx, image_fp in enumerate(images)
        ]
        add_images_to_napari(viewer, images, image_names)
    if labels is not None:
        if label_keys is None:
            label_keys = ['data'] * len(labels)
        labels = [
            _get_data(label_fp, label_keys[idx])
            for idx, label_fp in enumerate(labels)]
        add_labels_to_napari(viewer, labels, label_names)

    napari.run()
