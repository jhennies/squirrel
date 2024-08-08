
def morphological_operation_workflow(
        stack_path,
        out_path,
        key=key,
        pattern=pattern,
        operation=operation,
        verbose=verbose
):

    # Load the data
    from squirrel.library.io import load_data_handle
    data = load_data_handle(stack_path, key=key, pattern=pattern)[0][:]

    # Perform the operation
    result = func(data)

    # Save the result
    from squirrel.library.io import write_stack
    write_stack(out_path, result, 'data')
