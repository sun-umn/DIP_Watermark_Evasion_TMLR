import torch, cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from .rp_utils import BNNet, tv1_loss

# === Project Import ===
from model_dip import get_net_dip
from utils.general import uint8_to_float, float_to_int, img_np_to_tensor, \
    tensor_output_to_image_np, watermark_np_to_str, compute_bitwise_acc

#===========================================================#
# === Global hyper-parameters for Random Projector ===#
#===========================================================#
TV_WEIGHT = 0.45
LR = 1e-1
DEPTH = 3
W = 512
H = 512


def get_model(dig_cfgs):
    if dig_cfgs["arch"] == "random_projector":
        dip_model = get_net_dip(dig_cfgs["arch"])
    else:
        raise RuntimeError("Unsupported DIP architecture.")
    dip_model.train()
    return dip_model


def rp_evasion_single_img(
    im_orig_path, im_w_path, watermarker, watermark_gt, dip_cfgs=None
):
    """
        im_orig_path --- path to orig image
        im_w_path --- path to watermarked image
        watermarker --- a watermarker: watermark_decoded = watermarker.decode(im_w_np_bgr)
        watermark_gt --- ndarray with shape (n,), ground truth watermark
        dip_cfgs --- a dict of dip config params
        save_interm --- if set True, will return a list of reconstructed images of the intermediate steps
        detection_threshold --- the threshold of bitwise acc. to determin whether the image is watermarked.
    """
    assert dip_cfgs is not None, "Must include configs of the dip evation algo."
    device = dip_cfgs["device"]
    dtype = dip_cfgs["dtype"]
    save_interms = dip_cfgs["save_interms"]
    detection_threshold = dip_cfgs['detection_threshold']
    verbose = dip_cfgs["verbose"]

    watermark_gt_str = watermark_np_to_str(watermark_gt)

    # Init a random projector model
    dip_model  = get_model(dip_cfgs).to(device, dtype=dtype)
    show_every = dip_cfgs["show_every"]
    total_iters = dip_cfgs["total_iters"]
    
    #===========================================================#
    # Create a trainable batch norm and input seed
    #===========================================================#
    bnnet = BNNet(DEPTH)
    noise_like = torch.empty(1, DEPTH, W, H)
    g_noise = torch.zeros_like(noise_like).normal_() * 1e-1
    g_noise.requires_grad = True

    #===========================================================#
    # Define Optimizer
    #===========================================================#
    params = []
    params += dip_model.parameters()
    params += bnnet.parameters()
    params += [g_noise]
    optimizer = torch.optim.Adam(params, lr=LR)
    loss_func = torch.nn.MSELoss()

    # Optimize
    im_w_uint8_bgr = cv2.imread(im_w_path)
    im_orig_uint8_bgr = cv2.imread(im_orig_path)
    im_orig_bgr_float = uint8_to_float(im_orig_uint8_bgr)
    im_w_bgr_float = uint8_to_float(im_w_uint8_bgr)
    im_w_bgr_tensor = img_np_to_tensor(im_w_bgr_float).to(device, dtype=dtype)
    
    # Compute MSE of im_orig v.s. im_w
    mse_orig_w = np.mean((im_orig_bgr_float - im_w_bgr_float) ** 2)
    print("Im-W and Im-Orig MSE: {:.06f}".format(mse_orig_w))

    # Prepare logs for intermediate results
    iter_log = []
    bitwise_acc_log = []
    mse_clean_log = []
    psnr_clean_log = []
    mse_w_log = []
    psnr_w_log = []
    recon_interm_log = []  # saves the iterm recon result
    best_iter, best_psnr, best_mse = 0, -float("inf"), -float("inf")

    for num_iter in range(total_iters):
        optimizer.zero_grad()
        #===========================================================#
        # Compute Loss and Update 
        #===========================================================#
        g_noise_input = bnnet(g_noise)
        net_output = dip_model(g_noise_input)
        total_loss = loss_func(net_output, im_w_bgr_tensor)
        total_loss = torch.sqrt(total_loss)
        total_loss = total_loss + TV_WEIGHT * tv1_loss(net_output)
        total_loss.backward()
        optimizer.step()

        # Log Interm Result
        if num_iter % show_every == 0:
            iter_log.append(num_iter)

            img_recon = tensor_output_to_image_np(net_output)
            img_recon_np_int = float_to_int(img_recon)
            if save_interms:
                recon_interm_log.append(img_recon_np_int.astype(np.uint8))

            # Compute component-wise mse (using numpy array in the float scale [0, 1])
            mse_to_clean = np.mean((img_recon - im_orig_bgr_float)**2)
            mse_watermark = np.mean((img_recon - im_w_bgr_float)**2)
            mse_clean_log.append(mse_to_clean)
            mse_w_log.append(mse_watermark)

            # Compute PSNR
            psnr_recon_w = compute_psnr(
                im_w_uint8_bgr.astype(np.int16), img_recon_np_int, data_range=255  # PSNR of recon v.s. watermarked img
            )
            psnr_recon_orig = compute_psnr(
                im_orig_uint8_bgr.astype(np.int16), img_recon_np_int, data_range=255  # PSNR of recon v.s. orig
            )
            psnr_w_log.append(psnr_recon_w)
            psnr_clean_log.append(psnr_recon_orig)

            # Decode the recon image and compute the bitwise acc.
            img_recon_bgr_int8 = img_recon_np_int.astype(np.uint8)
            watermark_recon = watermarker.decode(img_recon_bgr_int8)
            watermark_recon_str = watermark_np_to_str(watermark_recon)
            bitwise_acc = compute_bitwise_acc(watermark_gt, watermark_recon)
            bitwise_acc_log.append(bitwise_acc)

            # Update the best recon result
            if psnr_recon_w > best_psnr and bitwise_acc < detection_threshold:
                best_iter = num_iter
                best_psnr = psnr_recon_w
                best_mse = mse_watermark
            
            if verbose:
                print("===== Iter [%d] - Loss [%.06f] =====" % (num_iter, total_loss.item()))
                print("  PSNR - <recon v.s im_w> %.02f | <recon v.s clean> %.02f " % (psnr_recon_w, psnr_recon_orig))
                print("  Recon Bitwise Acc. - {:.4f} % ".format(bitwise_acc * 100))
                print("Watermarks: ")
                print("GT:    {}".format(watermark_gt_str))
                print("Recon: {}".format(watermark_recon_str))
    
    return_log = {
        "iter_log": iter_log,
        "psnr_w": psnr_w_log,
        "psnr_clean": psnr_clean_log,
        "bitwise_acc": bitwise_acc_log,
        "interm_recon": recon_interm_log,
        "best_evade_iter": best_iter,
        "best_evade_psnr": best_psnr,
        "best_evade_mse": best_mse,
        "mse_to_orig": mse_clean_log,
        "mse_to_watermark": mse_w_log,
        "mse_clean_to_w": mse_orig_w
    }
    return return_log


if __name__ == "__main__":
    print("Unit test goes here.")
    