import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)

import argparse, pickle, os, cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from utils.general import compute_ssim, save_image_bgr
import pandas as pd


THRESHOLDS_DICT = {
    1: 0.55,
    2: 0.65,
    3: 0.75,
    4: 0.85,
    5: 0.95
}
def main(args):
    data_root_dir = os.path.join(
        "Result-Decoded", args.watermarker, args.dataset, args.evade_method, args.arch
    )
    file_names = [f for f in os.listdir(data_root_dir) if ".pkl" in f]
    for file_name in file_names:
        
        # Load Data
        file_path = os.path.join(data_root_dir, file_name)
        with open(file_path, 'rb') as handle:
            data_dict = pickle.load(handle)

        index_log = data_dict["index"]
        watermark_gt_str = data_dict["watermark_gt_str"]
        watermark_decoded_str = data_dict["watermark_decoded"]

        # 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, 
        help="Specification of watermarking method. [rivaGan, dwtDctSvd]",
        default="StegaStamp"
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, 
        help="Dataset [COCO, DiffusionDB]",
        default="COCO"
    )
    parser.add_argument(
        "--evade_method", dest="evade_method", type=str, help="Specification of evasion method.",
        default="vae"
    )
    parser.add_argument(
        "--arch", dest="arch", type=str, 
        help="""
            Secondary specification of evasion method (if there are other choices).

            Valid values a listed below:
                dip --- ["vanila", "random_projector"],
                vae --- ["cheng2020-anchor", "mbt2018", "bmshj2018-factorized"],
                corrupters --- ["gaussian_blur", "gaussian_noise", "bm3d", "jpeg", "brightness", "contrast"]
                diffuser --- Do not need.
                diffpure --- ["0.1", "0.2", "0.3"]
        """,
        default="cheng2020-anchor"
    )
    args = parser.parse_args()

    main(args)

    print("Completed")