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

    # === Read in watermarked images ===
    im_w_path = os.path.join("dataset", watermarker, dataset, "encoder_img", im_name)
    im_w_bgr_uint8 = cv2.imread(im_w_path)
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
    center = 255.5
    total_dim = 512

    for img_idx in range(0, len(interm_fft_log), 10):
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
    fig, ax = plt.subplots(ncols=1, nrows=1)
    for i in range(5):
        plt.plot(avg_err[:, i], label="Band {}".format(i))
    plt.legend()
    plt.show()
    # =======================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--dataset", dest="dataset", type=str, help="Dataset name.",
        default="COCO"
    )
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, help="Watermarker Name.",
        default="rivaGan"
    )
    parser.add_argument(
        "--im_name", dest="im_name", type=str, help="Clean image name.",
        default="Img-10.png"
    )
    parser.add_argument(
        "--dip_arch", dest="dip_arch", type=str, help="DIP arch.",
        default="vanila"
    )
    args = parser.parse_args()
    main(args)

    print(" ****** Completed ****** ")