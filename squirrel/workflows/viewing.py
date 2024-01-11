
def view_in_napari(
        images=None,
        labels=None,
        image_keys=None,
        label_keys=None,
        image_names=None,
        label_names=None,
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
    from squirrel.library.io import load_data

    viewer = napari.Viewer()

    if images is not None:
        if image_keys is None:
            image_keys = ['data'] * len(images)
        images = [load_data(image_fp, image_keys[idx]) for idx, image_fp in enumerate(images)]
        add_images_to_napari(viewer, images, image_names)
    if labels is not None:
        if label_keys is None:
            label_keys = ['data'] * len(labels)
        labels = [load_data(label_fp, label_keys[idx]) for idx, label_fp in enumerate(labels)]
        add_labels_to_napari(viewer, labels, label_names)

    napari.run()
