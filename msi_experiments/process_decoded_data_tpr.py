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


def calc_bitwise_acc(gt_str, decoded_str):
    correct, total = 0., 0.
    for i in range(min(len(gt_str), len(decoded_str))):
        if gt_str[i] == decoded_str[i]:
            correct = correct + 1.
        total = total + 1.
    if len(gt_str) != len(decoded_str):
        total += abs(len(gt_str) - len(decoded_str))
    return correct / total


# THRESHOLDS_DICT = {
#     1: 0.55,
#     2: 0.65,
#     3: 0.75,
#     4: 0.85,
#     5: 0.95
# }
THRESHOLDS_DICT = {
    1: 0.55,
    2: 0.65,
    3: 0.75,
    4: 0.85,
}

def main(args):
    print("Watermarker: ", args.watermarker)
    summary_file = os.path.join(".", "dataset", args.watermarker, args.dataset, "water_mark.csv")
    annot_data = pd.read_csv(summary_file)

    len_data = len(annot_data)
    print("Total Data Len: ", len_data)
    watermark_gt_str = annot_data.iloc[0]["Encoder"]
    if watermark_gt_str[0] == "[":  # Some historical none distructive bug :( will cause this reformatting
        watermark_gt_str = eval(watermark_gt_str)[0]

    res_dict = {}
    for key in THRESHOLDS_DICT.keys():
        res_dict[key] = []


    for idx in range(len_data):
        data = annot_data.iloc[idx]
        watermark_decoded_str = data["Decoder"]
        if watermark_decoded_str[0] == "[":  # Some historical none distructive bug :( will cause this reformatting
            watermark_decoded_str = eval(watermark_decoded_str)[0]
        ba = calc_bitwise_acc(watermark_gt_str, watermark_decoded_str)
        for key in THRESHOLDS_DICT.keys():
            value = THRESHOLDS_DICT[key]
            if ba >= value:
                res_dict[key].append(1)
            else:
                res_dict[key].append(0)

    for key in res_dict.keys():
        print("Threshold: ", THRESHOLDS_DICT[key])
        print("  TPR: ", np.mean(res_dict[key]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, 
        help="Specification of watermarking method. [rivaGan, dwtDctSvd]",
        default="rivaGan"
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, 
        help="Dataset [COCO, DiffusionDB]",
        default="COCO"
    )
    args = parser.parse_args()

    main(args)

    print("Completed")