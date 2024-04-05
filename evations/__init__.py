from .dip_evation import dip_evasion_single_img


def get_evasion_alg(evade_method):
    if evade_method.lower() == "dip":
        method = dip_evasion_single_img
        print("Initiated a ***{}*** evader. ".format(evade_method))
    else:
        raise RuntimeError("Unsupported evasion method specified")
    return method