from .dip_evation import dip_evasion_single_img
from .vae_evation import vae_evasion_single_img
from .corrupters import corruption_evation_single_img


def get_evasion_alg(evade_method):
    if evade_method.lower() == "dip":
        method = dip_evasion_single_img
    elif evade_method.lower() == "vae":
        method = vae_evasion_single_img
    elif evade_method.lower() == "corrupters":
        method = corruption_evation_single_img
    else:
        raise RuntimeError("Unsupported evasion method specified")
    print("Initiated a ***{}*** evader. ".format(evade_method))
    return method