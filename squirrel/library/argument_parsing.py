
def column_separated_inputs_to_dict(inputs):

    if inputs is None:
        return None

    if type(inputs) == list:
        out_dict = dict()
        for inp in inputs:
            key, values = str.split(inp, ':')
            values = str.split(values, ',')
            out_dict[key] = values

        return out_dict

    raise TypeError(f'Inputs must be either None or list of string. Found type(inputs) = {type(inputs)}')
