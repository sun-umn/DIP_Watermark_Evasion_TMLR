import torch
from model_dip import get_net_dip
from utils.general import uint8_to_float, float_to_int, img_np_to_tensor


def get_model(dig_cfgs):
    if dig_cfgs.arch == "vanila":
        dip_model = get_net_dip()
    else:
        raise RuntimeError("Unsupported DIP architecture.")



def dip_evasion_single_img(
    im_orig_uint8_bgr, im_w_unit8_bgr, decoder, watermark_gt, dip_cfgs, 
    device=torch.device("cuda"), dtype=torch.float
):
    """
        im_orig_np_bgr --- ndarray with shape (N, N 3) | dtype = np.uint8 | value range - [0, 255]
        im_w_np_bgr --- ndarray with shape (N, N, 3) | dtype = np.uint8 | value range - [0, 255]
        decoder --- a decoder that: watermark_decoded = decoder.decode(im_w_np_bgr)
        watermark_gt --- ndarray with shape (n,), ground truth watermark
        dip_cfgs --- a dict of dip config params
    """
    res_dict = {}  # A dictionary to return intermediate reconstruction

    # Init A DIP model
    dip_model  = get_model(dip_cfgs).to(device, dtype=dtype)
    dip_model.train()
    show_every = dip_cfgs["show_every"]
    total_iters = dip_cfgs["total_iters"]
    lr = dip_cfgs["lr"]
    params = dip_model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_func = torch.nn.MSELoss()

    # Optimize
    im_w_bgr_float = uint8_to_float(im_w_unit8_bgr)
    im_w_bgr_tensor = img_np_to_tensor(im_w_bgr_float).to(device, dtype=dtype)
    
    iter_log = []
    bitwise_acc_log = []
    psnr_w_log = []
    psnr_clean_log = []




if __name__ == "__main__":
    print("Unit test goes here.")