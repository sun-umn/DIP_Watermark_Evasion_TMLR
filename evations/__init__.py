from .dip_evation import dip_evasion_single_img, dip_interm_collection
from .rp_evation import rp_evasion_single_img
from .vae_evation import vae_evasion_single_img
from .corrupters import corruption_evation_single_img
from .diffusion_evation import diffuser_evation_single_img


def get_evasion_alg(method_name, arch=None):
    if method_name.lower() == "dip":
        if arch.lower() == "vanila":
            method = dip_evasion_single_img
        elif arch.lower() == "random_projector":
            method = rp_evasion_single_img
        else:
            raise RuntimeError("Unsupported DIP arch specified")
    elif method_name.lower() == "vae":
        method = vae_evasion_single_img
    elif method_name.lower() == "corrupters":
        method = corruption_evation_single_img
    elif method_name.lower() == "diffuser":
        method = diffuser_evation_single_img
    else:
        raise RuntimeError("Unsupported evasion method specified")
    print("Initiated a ***{}*** evader. ".format(method_name))
    return method


def get_interm_collection_algo(method_name):
    if method_name.lower() == "dip":
        method = dip_interm_collection
    else:
        raise RuntimeError("Unsupported evasion method specified")
    
    print("Initiated a ***{}*** interm. data collection process. ".format(method_name))
    return method