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


def main(args):
    dataset = args.dataset
    watermarker = args.watermarker
    im_name = args.im_name
    pkl_name = args.im_name.replace(".png", ".pkl")

    # === Read in watermarked images ===
    im_w_path = os.path.join("dataset", watermarker, dataset, im_name)
    im_w_bgr_uint8 = cv2.imread(im_w_path)
    im_w_gray_unit8 = cv2.cvtColor(im_w_bgr_uint8, cv2.COLOR_BGR2GRAY)
    im_w_bgr_int = im_w_bgr_uint8.astype(np.int32)
    im_w_gray_int = im_w_gray_unit8.astype(np.int32)
    im_w_bgr_float = uint8_to_float(im_w_bgr_uint8)

    dip_recon_path = os.path.join(
        "Result-Interm", watermarker, "dip", args.dip_arch, pkl_name
    )




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