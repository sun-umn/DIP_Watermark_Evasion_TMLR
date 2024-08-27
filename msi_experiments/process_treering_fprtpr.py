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
    1: 20,
    2: 40,
    3: 60,
    4: 80,
}

def main(args):
    print("Watermarker: ", args.watermarker)
    summary_file = os.path.join(".", "dataset", args.watermarker, args.dataset, "water_mark.csv")
    annot_data = pd.read_csv(summary_file)

    len_data = len(annot_data)
    print("Total Data Len: ", len_data)

    res_dict_tpr = {}
    res_dict_fpr = {}
    for key in THRESHOLDS_DICT.keys():
        res_dict_tpr[key] = []
        res_dict_fpr[key] = []


    for idx in range(len_data):
        data = annot_data.iloc[idx]
        clean_distance = data["Decode_Clean"]
        w_distance = data["Decode_W"]
        for key in THRESHOLDS_DICT.keys():
            value = THRESHOLDS_DICT[key]
            if clean_distance >= value:
                res_dict_fpr[key].append(1)
            else:
                res_dict_fpr[key].append(0)
            
            if w_distance <= value:
                res_dict_tpr[key].append(1)
            else:
                res_dict_tpr[key].append(0)

    for key in res_dict_tpr.keys():
        print("Threshold: ", THRESHOLDS_DICT[key])
        print("  TPR: ", np.mean(res_dict_tpr[key]))
        print("  FPR: ", np.mean(res_dict_fpr[key]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, 
        help="Specification of watermarking method. [rivaGan, dwtDctSvd]",
        default="Tree-Ring"
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, 
        help="Dataset [COCO, DiffusionDB]",
        default="Gustavosta"
    )
    args = parser.parse_args()

    main(args)

    print("Completed")