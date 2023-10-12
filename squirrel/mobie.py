
def merge_mobie_projects(
        mobie_a,
        mobie_b,
        out_folder,
        datasets_a=None,
        datasets_b=None,
        copy_data=False,
        verbose=False
):

    # FIXME actually this functionality is way better off in the mobie_utils repo!

    # TODO: This function should check for the following:
    #  - The input mobie projects must exist
    #  - The datasets must exist in the respective mobie projects
    #  - There must be no datasets of equal name in both inputs
    if not inputs_are_valid([mobie_a, mobie_b], [datasets_a, datasets_b]):
        print(f'Error: Non-valid inputs!')
        return

    make_empty_output_dataset(out_folder)

    populate_output_dataset(mobie_a, datasets_a, copy_data=copy_data)
    populate_output_dataset(mobie_b, datasets_b, copy_data=copy_data)
