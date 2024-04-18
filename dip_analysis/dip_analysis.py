"""
    A script tries to explore why DIP can work in this context.
"""

import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import math, torch, cv2, argparse
import numpy as np

# === Project Import ===
from watermarkers import get_watermarkers
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from evations import get_evasion_alg
from utils.plottings import plot_dip_res, plot_vae_res, plot_corruption_res, \
    plot_diffuser_res
from utils.general import watermark_np_to_str, uint8_to_float, img_np_to_tensor, \
    float_to_int, set_random_seeds
from model_dip import get_net_dip
import matplotlib.pyplot as plt


def get_model(dig_cfgs):
    if dig_cfgs["arch"] == "vanila":
        dip_model = get_net_dip()
    else:
        raise RuntimeError("Unsupported DIP architecture.")
    dip_model.train()
    return dip_model



def dip_regress_single_input(
    cfg, input_tensor, clip_min=0, clip_max=1,
    im_orig_tensor=None, w_tensor=None
):
    # Load Setups
    device = cfg["device"]
    dtype = cfg["dtype"]
    detection_threshold = cfg['detection_threshold']
    total_iters = cfg["total_iters"]
    lr = cfg["lr"]
    show_every = cfg["show_every"]
    verbose = cfg["verbose"]

    # === Regress im_w and trace the error from im_clean and noise ===
    dip_model  = get_model(cfg).to(device, dtype=dtype)
    params = dip_model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_func = torch.nn.MSELoss()
    iter_log = []
    mse_log = []
    orig_mse_log = []
    w_mse_log = []

    net_input = input_tensor.to(device)
    orig_tensor = im_orig_tensor.to(device)
    w_tensor = w_tensor.to(device)

    for num_iter in range(total_iters):
        optimizer.zero_grad()
        net_output = dip_model(net_input)

        # Compute Loss and Update 
        total_loss = loss_func(net_output, net_input)
        total_loss.backward()
        optimizer.step()
        
        # Log Interm Result
        if num_iter % show_every == 0:
            iter_log.append(num_iter)

            img_recon = np.transpose(torch.clamp(net_output.detach().cpu(), clip_min, clip_max).numpy()[0, :, :, :], [1, 2, 0])
            img_recon_np_int = float_to_int(img_recon)
            # recon_interm_log.append(img_recon_np_int.astype(np.uint8))

            # record training error
            mse_log.append(total_loss.item())

            with torch.no_grad():
                orig_mse = torch.nn.functional.mse_loss(net_output, orig_tensor).item()
                w_mse = torch.nn.functional.mse_loss(net_output-orig_tensor, w_tensor).item()
                orig_mse_log.append(orig_mse)
                w_mse_log.append(w_mse)

            if verbose:
                print("===== Iter [%d] - Loss [%.06f] =====" % (num_iter, total_loss.item()))
    res_log = {
        "iter_log": iter_log,
        "mse_log": mse_log,
        "orig_mse": orig_mse_log,
        "w_mse": w_mse_log
    }
    return res_log


def main(args):
    # === Some Dummy Configs ===
    device = torch.device("cuda")
    set_random_seeds(args.random_seed)
    
    img_clean_path = os.path.join(
        args.root_path_im_orig, args.im_name  # Path to a clean image
    )

    img_w_root_dir = os.path.join(
        args.root_path_im_w, args. watermarker
    )
    os.makedirs(img_w_root_dir, exist_ok=True)
    img_w_path = os.path.join(
        img_w_root_dir, args.im_name  # Path to save the watermarked image.
    )

    # === Initiate a watermark ==> in ndarray
    watermark_gt = np.random.binomial(1, 0.5, 32)  

    # === Initiate a encoder & decoder ===
    watermarker_configs = {
        "watermarker": args.watermarker,
        "watermark_gt": watermark_gt
    }
    watermarker = get_watermarkers(watermarker_configs)

    # Generated watermarked image and save it to img_w_path
    watermarker.encode(img_clean_path, img_w_path) 
    # Check decoding in case learning-based encoder/decoder doesn't work properly
    watermark_decode = watermarker.decode_from_path(img_w_path)
    bitwise_acc_0 = np.mean(watermark_decode == watermark_gt)
    print("*Sanity check for watermarker encoder & decoder:")
    print("  Decoded watermark from im_w: {}".format(watermark_np_to_str(watermark_decode)))
    print("  Bitwise acc. - [{:.04f} %]".format(bitwise_acc_0 * 100))
    assert bitwise_acc_0 > 0.99, "The encoder & decode fails to work on this watermark string."
    
    # Read configs and execude evasions
    detection_threshold = args.detection_threshold
    print("Setting detection threshold [{:02f}] for the watermark detector.".format(detection_threshold))
    CONFIGS = {
        "arch": "vanila",   # Used in DIP to select the variant architecture
        "show_every": 1,   # Used in DIP to log interm. result
        "total_iters": 500, # Used in DIP as the max_iter
        "lr": 0.01,         # Used in DIP as the learning rate

        "device": device,
        "dtype": torch.float,
        "detection_threshold": detection_threshold,
        "verbose": True,
        "save_interms": True
    }

    # ==== Create log folder ====
    vis_root_dir = os.path.join(
        ".", "Vis-Explore", "{}".format(args.im_name.split(".")[0]), "{}".format(args.watermarker), "{}".format("dip"), "{}".format(CONFIGS["arch"])
    )
    os.makedirs(vis_root_dir, exist_ok=True)

    # ==== Setup the experiment ===
    im_w_uint8_bgr = cv2.imread(img_w_path)
    im_orig_uint8_bgr = cv2.imread(img_clean_path)
    im_residual_int_bgr = im_w_uint8_bgr.astype(np.int16) - im_orig_uint8_bgr.astype(np.int16)
    print("Sanity check for residual calculation: ", np.amin(im_residual_int_bgr), np.amax(im_residual_int_bgr))
    
    # im_res_bgr_float = uint8_to_float(im_residual_int_bgr)
    # im_res_bgr_tensor = img_np_to_tensor(im_res_bgr_float)
    
    # # Regress clean img
    im_orig_bgr_float = uint8_to_float(im_orig_uint8_bgr)
    im_orig_bgr_tensor = img_np_to_tensor(im_orig_bgr_float)
    w_bgr_float = uint8_to_float(im_residual_int_bgr)
    w_bgr_tensor = img_np_to_tensor(w_bgr_float)
    # res_orig = dip_regress_single_input(
    #     CONFIGS, im_orig_bgr_tensor, 0, 1
    # )

    # Regress im_w
    im_w_bgr_float = uint8_to_float(im_w_uint8_bgr)
    im_w_bgr_tensor = img_np_to_tensor(im_w_bgr_float)
    res_w = dip_regress_single_input(
        CONFIGS, im_w_bgr_tensor, 0, 1,
        im_orig_bgr_tensor, w_bgr_tensor
    )

    # # Regress w
    # w_bgr_float = uint8_to_float(im_residual_int_bgr)
    # w_bgr_tensor = img_np_to_tensor(w_bgr_float)
    # res_w_mark = dip_regress_single_input(
    #     CONFIGS, w_bgr_tensor, 0, 1
    # )

    # == Plot res ==
    fig, ax = plt.subplots(ncols=1, nrows=1)
    l1 = ax.plot(res_w["iter_log"], res_w["orig_mse"], label="Clean Img MSE")
    l2 = ax.plot(res_w["iter_log"], res_w["w_mse"], label="Watermark MSE")
    # l3 = ax.plot(res_w_mark["iter_log"], res_w_mark["mse_log"], label="Watermark Only")
    ax.set_xscale('log')
    ax.legend()
    save_name = os.path.join(vis_root_dir, "MSE_plot.png")
    plt.savefig(save_name)
    plt.close(fig)



if __name__ == "__main__": 

    print("\n***** This is demo of single image evasion ***** \n")
    
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--random_seed", dest="random_seed", type=int, help="Manually set random seed for reproduction.",
        default=42
    )
    parser.add_argument(
        '--root_path_im_orig', type=str, help="Root folder to the clean images.",
        default=os.path.join("examples", "ori_imgs")
    )
    parser.add_argument(
        "--im_name", dest="im_name", type=str, help="clean image name.",
        default="711.png"
    )
    parser.add_argument(
        "--root_path_im_w", dest="root_path_im_w", type=str, help="Root folder to save watermarked image.",
        default=os.path.join("examples", "watermarked_imgs")
    )
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, help="Specification of watermarking method.",
        default="rivaGan"
    )
    parser.add_argument(
        "--arch", dest="arch", type=str, help="Secondary specification of evasion method (if there are other choices).",
        default="bm3d"
    )
    parser.add_argument(
        "--detection_threshold", dest="detection_threshold", type=float, default=0.75,
        help="Tunable threhsold to check if the evasion is successful."
    )
    args = parser.parse_args()
    main(args)

    print("\n***** Completed. *****\n")
