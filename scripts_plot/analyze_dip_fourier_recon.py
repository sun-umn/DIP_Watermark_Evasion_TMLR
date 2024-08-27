import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import math, torch, cv2, argparse
from pytorch_msssim import ssim, ms_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# ==
from utils.general import watermark_np_to_str, uint8_to_float
from scripts_plot.analyze_watermark_fourier import calc_fft_three_channel


def compute_fft_band_err(fft_w, fft_interm):
    # |F(x)|
    fft_w_mag = np.absolute(fft_w)
    # |F(x) - F(\hat(x))|
    err_fft = fft_w - fft_interm
    fft_err_mag = np.absolute(err_fft)
    # |F(x) - F(\hat(x))| / |F(x)|
    pointwise_err = fft_err_mag / fft_w_mag
    return pointwise_err


def main(args):
    dataset = args.dataset
    watermarker = args.watermarker
    im_name = args.im_name
    pkl_name = args.im_name.replace(".png", ".pkl")
    if watermarker == "StegaStamp":
        im_name = im_name.replace(".png", "_hidden.png")

    # === Read in watermarked images ===
    im_w_path = os.path.join("dataset", watermarker, dataset, "encoder_img", im_name)
    im_w_bgr_uint8 = cv2.imread(im_w_path)
    if watermarker == "StegaStamp":
        im_w_bgr_uint8 = cv2.resize(im_w_bgr_uint8, (512, 512), interpolation=cv2.INTER_CUBIC)

    im_w_bgr_int = im_w_bgr_uint8.astype(np.int32)
    im_w_bgr_float = uint8_to_float(im_w_bgr_uint8)
    # FFT of the watermarked image
    fourier_w = calc_fft_three_channel(im_w_bgr_float)

    dip_recon_path = os.path.join(
        "Result-Interm", watermarker, dataset, "dip", args.dip_arch, pkl_name
    )
    with open(dip_recon_path, 'rb') as handle:
        interm_data_dict = pickle.load(handle)
    dip_interm_recons = interm_data_dict["interm_recon"]
    dip_interm_indices = interm_data_dict["index"]
    
    interm_idx_log, interm_fft_log = [], []
    for idx, interm_recon_bgr_uint8 in enumerate(dip_interm_recons):
        if watermarker == "StegaStamp":
            interm_recon_bgr_uint8 = cv2.resize(interm_recon_bgr_uint8, (512, 512), interpolation=cv2.INTER_CUBIC)
        # Append index
        interm_idx_log.append(dip_interm_indices[idx])
        
        # Compute FFT of each interm recons
        interm_recon_bgr_float = uint8_to_float(interm_recon_bgr_uint8)
        fft_interm = calc_fft_three_channel(interm_recon_bgr_float)
        interm_fft_log.append(fft_interm)

    print("FFT computed. Now compute fft band error.")

    # === Calc mean fft band error ===
    fft_band_error_log = []
    fft_band_count_log = []
    index_log = []
    center = 255.5
    total_dim = 512

    for img_idx in range(0, len(interm_fft_log), 10):
        index_log.append(dip_interm_indices[img_idx])
        print("Processing {}".format(img_idx))
        interm_fft = interm_fft_log[img_idx]
        point_wise_err = compute_fft_band_err(fourier_w, interm_fft)

        fft_err_dict = np.zeros(5)
        fft_count_dict = np.zeros(5)

        for i in range(total_dim):
            for j in range(total_dim):
                err_value = np.sum(point_wise_err[:, i, j], axis=0)
                radius = int(np.floor(np.sqrt((i-center)**2 + (j-center)**2)))
                
                section = min(radius // 50, 4)
                
                fft_err_dict[section] += err_value
                fft_count_dict[section] += 3

        fft_band_error_log.append(fft_err_dict)
        fft_band_count_log.append(fft_count_dict)
    
    fft_band_error_log = np.asarray(fft_band_error_log)
    fft_band_count_log = np.asarray(fft_band_count_log)
    avg_err = fft_band_error_log / fft_band_count_log

    # === Plot the result ===
    # TBD: Make it better looking 
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 3))
    for i in range(5):
        if i == 0:
            msg = "{} (lowest)".format(i+1)
        elif i == 4:
            msg = "{} (highest)".format(i+1)
        else:
            msg = "{}".format(i+1)
        ax.plot(index_log, avg_err[:, i], label=msg, lw=2, alpha=0.8)
    # ax.legend(loc='upper right', ncol=1, fancybox=True, shadow=False, fontsize=15, framealpha=0.3)
    ax.set_xticks([0, 200, 400])
    ax.set_yticks([0.5, 1.0])
    ax.grid("both")
    ax.set_ylim([0, 1.2])
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=0)
    plt.tight_layout()

    vis_dir = os.path.join(".", "Vis-FBE")
    os.makedirs(vis_dir, exist_ok=True)
    save_name = os.path.join(vis_dir, "{}.png".format(watermarker))
    plt.savefig(save_name)
    plt.close(fig)
    # =======================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--dataset", dest="dataset", type=str, help="Dataset name.",
        default="COCO"
    )
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, help="Watermarker Name.",
        default="dwtDctSvd"
    )
    parser.add_argument(
        "--im_name", dest="im_name", type=str, help="Clean image name.",
        default="Img-1.png"
    )
    parser.add_argument(
        "--dip_arch", dest="dip_arch", type=str, help="DIP arch.",
        default="vanila"
    )
    args = parser.parse_args()
    main(args)

    print(" ****** Completed ****** ")