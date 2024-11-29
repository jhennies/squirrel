
def add_images_to_napari(viewer, images, names=None):
    if names is None:
        names = [f'image{idx}' for idx in range(len(images))]
    for img_idx, img in enumerate(images):
        viewer.add_image(img, name=names[img_idx])


def add_labels_to_napari(viewer, labels, names=None):
    if names is None:
        names = [f'label{idx}' for idx in range(len(labels))]
    for lbl_idx, lbl in enumerate(labels):
        # viewer.add_labels(lbl.astype('uint8'), name=names[lbl_idx])
        viewer.add_labels(lbl.astype('uint32'), name=names[lbl_idx])
