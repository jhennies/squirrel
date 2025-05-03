
def get_n5_handle(
        filepath,
        key=None,
        mode='r'
):

    from z5py import File
    if key is not None:
        return File(filepath, mode=mode)[key]
    return File(filepath, mode)
