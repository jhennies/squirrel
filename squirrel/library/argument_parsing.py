
def column_separated_inputs_to_dict(inputs):
    if inputs is None:
        return None

    if not isinstance(inputs, list):
        raise TypeError(
            f'Inputs must be either None or list of strings. Found type(inputs) = {type(inputs)}'
        )

    def parse_value(value):
        # try int
        try:
            return int(value)
        except ValueError:
            pass

        # try float
        try:
            return float(value)
        except ValueError:
            pass

        # fallback to string
        return value

    out_dict = {}
    for inp in inputs:
        key, value = inp.split(':', 1)  # split only on first colon
        out_dict[key] = parse_value(value)

    return out_dict
