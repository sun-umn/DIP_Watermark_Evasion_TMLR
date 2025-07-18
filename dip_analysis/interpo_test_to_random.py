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
    float_to_int, set_random_seeds, float_to_uint8,save_image_bgr, compute_bitwise_acc
from model_dip import get_net_dip
import matplotlib.pyplot as plt


def main(args):
    # === Some Dummy Configs ===
    use_random_type = "zeros"  # ["random", "zeros"]
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

    # ==== Create log folder ====
    vis_root_dir = os.path.join(
        ".", "Vis-Test-Linear", "{}".format(args.im_name.split(".")[0]), "{}".format(args.watermarker), "{}".format("interpo_linear_from_random"), "{}".format(use_random_type)
    )
    os.makedirs(vis_root_dir, exist_ok=True)

    # ==== Setup the experiment ===
    im_w_uint8_bgr = cv2.imread(img_w_path)
    im_orig_uint8_bgr = cv2.imread(img_clean_path)
    im_residual_int_bgr = im_w_uint8_bgr.astype(np.int16) - im_orig_uint8_bgr.astype(np.int16)
    print("Sanity check for residual calculation: ", np.amin(im_residual_int_bgr), np.amax(im_residual_int_bgr))
    
    # Convert the images to float 
    im_orig_bgr_float = uint8_to_float(im_orig_uint8_bgr)
    im_w_bgr_float = uint8_to_float(im_w_uint8_bgr)
    mse_clean_to_w = np.mean((im_orig_bgr_float-im_w_bgr_float)**2)
    # Generate a random init point
    if use_random_type == "random":
        random_start_float = np.random.random_sample(size=im_w_bgr_float.shape)
    elif use_random_type == "zeros":
        random_start_float = np.zeros_like(im_w_bgr_float)
    print(im_w_bgr_float.shape, random_start_float.shape)
    
    # === Vis random start for sanity check ===
    random_start_uint8 = float_to_uint8(random_start_float)
    save_name = os.path.join(vis_root_dir, "image_random_init.png")
    save_image_bgr(random_start_uint8, save_name)

    # === Generate search candidates ===
    ratios = list(np.arange(1, 100, 1) * 0.01)
    # Check the interpo. result
    ratio_log = []
    bitwise_acc_log = []
    mse_clean_log = []
    psnr_clean_log = []
    mse_w_log = []
    psnr_w_log = []
    recon_interm_log = []  # saves the iterm recon result
    best_idx, best_ratio, best_psnr, best_mse = 0, 0, -float("inf"), -float("inf")
    for idx, ratio in enumerate(ratios):
        im_interm_float = np.clip((ratio) * random_start_float + (1-ratio) * im_w_bgr_float, 0, 1)
        im_interm_uint8 = float_to_uint8(im_interm_float)
        
        ratio_log.append(ratio)
        recon_interm_log.append(im_interm_uint8)

        # Calc Quality
        mse_clean = np.mean((im_orig_bgr_float - im_interm_float)**2)
        psnr_clean = compute_psnr(
            im_orig_uint8_bgr.astype(np.int16),
            im_interm_uint8.astype(np.int16),
            data_range=255
        )
        mse_w = np.mean((im_w_bgr_float - im_interm_float)**2)
        psnr_w = compute_psnr(
            im_w_uint8_bgr.astype(np.int16),
            im_interm_uint8.astype(np.int16),
            data_range=255
        )
        mse_clean_log.append(mse_clean)
        psnr_clean_log.append(psnr_clean)
        mse_w_log.append(mse_w)
        psnr_w_log.append(psnr_w)

        # Calc decoded string
        watermark_recon = watermarker.decode(im_interm_uint8)
        watermark_recon_str = watermark_np_to_str(watermark_recon)
        bitwise_acc = compute_bitwise_acc(watermark_gt, watermark_recon)
        bitwise_acc_log.append(bitwise_acc)

        # Update the best recon result
        if psnr_w > best_psnr and bitwise_acc < detection_threshold:
            best_idx = idx
            best_ratio = ratio
            best_psnr = psnr_clean
            best_mse = mse_clean

        print("===== Ratio [{:02f}] =====".format(ratio))
        print("  PSNR-w -  {:.04f} | PSNR-clean - {:.04f}".format(psnr_w, psnr_clean))
        print("  Recon Bitwise Acc. - {:.4f} % ".format(bitwise_acc * 100))
        print("Watermarks: ")

    print("==== Best ====")
    print(best_ratio, best_mse, best_psnr)
    res_log = {
        "ratios": ratio_log, 
        "mse_to_orig": mse_clean_log,
        "mse_to_watermark": mse_w_log,
        "psnr_clean": psnr_clean_log,
        "psnr_w": psnr_w_log,
        "bitwise_acc": bitwise_acc_log,
        "interm_recon": recon_interm_log,
        "best_evade_ratio": best_ratio,
        "best_evade_mse": best_mse,
        "best_evade_psnr": best_psnr
    }


    # ==== Vis Result ===
    # Plot Iter-PSNR curves and bitwise acc.
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ratio_data = res_log["ratios"]
    bw_acc_data = res_log["bitwise_acc"]
    psnr_w_data, psnr_clean_data = res_log["psnr_w"], res_log["psnr_clean"]
    ax[0].plot(ratio_data, psnr_clean_data, label="PSNR (recon - clean)", color="orange")
    ax[0].plot(ratio_data, psnr_w_data, label="PSNR (recon - watermarked)", color="blue", ls="dashed")
    ax[0].vlines(res_log["best_evade_ratio"], ymin=np.amin(psnr_w_data), ymax=np.amax(psnr_w_data), color="black", ls="dashed", label="Best Recon Iter")
    ax[0].legend()
    ax[1].plot(ratio_data, bw_acc_data, label="Bitwise Acc.")
    ax[1].hlines(y=detection_threshold, xmin=np.amin(ratio_data), xmax=np.amax(ratio_data), ls="dashed", color="black")
    ax[1].hlines(y=(1-detection_threshold), xmin=np.amin(ratio_data), xmax=np.amax(ratio_data), ls="dashed", color="black")
    ax[1].vlines(res_log["best_evade_ratio"], ymin=0, ymax=1, color="black", ls="dashed", label="Best Recon Iter")
    ax[1].legend()
    plt.tight_layout()
    save_name = os.path.join(vis_root_dir, "psnr_bt_acc.png")
    plt.savefig(save_name)
    plt.close(fig)

    # Vis iter-mse
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(ratio_data, res_log["mse_to_orig"], label="MSE (recon - clean)")
    ax[0].plot(ratio_data, res_log["mse_to_watermark"], label="MSE (recon - im_w)")
    ax[0].hlines(res_log["best_evade_mse"], xmin=0, xmax=np.amax(ratio_data), label="Best evade MSE (recon v.s. im_w)", ls="dashed", color="orange")
    ax[0].hlines(mse_clean_to_w, xmin=0, xmax=np.amax(ratio_data), label="MSE (clean v.s. im_w)", ls="dashed", color="black")
    ax[0].vlines(res_log["best_evade_ratio"], ymin=0, ymax=np.amax(res_log["mse_to_watermark"]), color="black", ls="dashed", label="Best Recon Iter")
    ax[0].legend()
    ax[0].set_yscale('log')
    ax[1].plot(ratio_data, bw_acc_data, label="Bitwise Acc.")
    ax[1].hlines(y=detection_threshold, xmin=np.amin(ratio_data), xmax=np.amax(ratio_data), ls="dashed", color="black")
    ax[1].hlines(y=(1-detection_threshold), xmin=np.amin(ratio_data), xmax=np.amax(ratio_data), ls="dashed", color="black")
    ax[1].vlines(res_log["best_evade_ratio"], ymin=0, ymax=1, color="black", ls="dashed", label="Best Recon Iter")
    ax[1].legend()
    save_name = os.path.join(vis_root_dir, "MSE_plot.png")
    plt.savefig(save_name)
    plt.close(fig)

    # Vis best evade recon image
    best_recon_image = res_log["interm_recon"][best_idx]
    best_ratio = res_log["best_evade_ratio"]
    print("Sanity check best ratio", best_ratio)
    save_name = os.path.join(vis_root_dir)


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
