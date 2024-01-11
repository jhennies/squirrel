
def main():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Show images and label maps in napari',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', '--images', nargs='+', type=str, default=None,
                        help='List of image files (h5 or nii)')
    parser.add_argument('-l', '--labels', nargs='+', type=str, default=None,
                        help='List of label map files (h5 or nii)')
    parser.add_argument('-ik', '--image_keys', nargs='+', default=None,
                        help='List of keys of dataset within the image files; '
                             'Required if --images are h5 files; Defaults to ["data"] * len(images)')
    parser.add_argument('-lk', '--label_keys', nargs='+', default=None,
                        help='List of keys of dataset within the label files; '
                             'Required if --labels are h5 files; Defaults to ["data"] * len(labels)')
    parser.add_argument('-in', '--image_names', nargs='+', type=str, default=None,
                        help='List of names of the image maps; Must be same length as --images; '
                             'If not supplied default names are used')
    parser.add_argument('-ln', '--label_names', nargs='+', type=str, default=None,
                        help='List of names of the label maps; Must be same length as --labels; '
                             'If not supplied default names are used')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    images = args.images
    labels = args.labels
    image_keys = args.image_keys
    label_keys = args.label_keys
    image_names = args.image_names
    label_names = args.label_names
    verbose = args.verbose

    from squirrel.workflows.viewing import view_in_napari

    view_in_napari(
        images=images,
        labels=labels,
        image_keys=image_keys,
        label_keys=label_keys,
        image_names=image_names,
        label_names=label_names,
        verbose=verbose
    )


if __name__ == '__main__':
    main()
