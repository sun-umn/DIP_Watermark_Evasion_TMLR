"""
    This script is use dwtDctSvd/rivaGan to decode clean images to compute FPR.
"""
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import os, argparse, torch
import cv2
import numpy as np
import pandas as pd
from utils.general import rgb2bgr, save_image_bgr, set_random_seeds, \
    watermark_np_to_str, watermark_str_to_numpy
from watermarkers import get_watermarkers


def main(args):
    # === Some dummt configs ===
    device = torch.device("cuda")
    set_random_seeds(args.random_seed)

    # === Set the dataset paths ===
    dataset_input_path = os.path.join(
        args.clean_data_root, args.dataset
    )
    output_root_path = os.path.join(
        ".", "dataset", "Clean_Watermark_Evasion", args.watermarker, args.dataset
    )
    os.makedirs(output_root_path, exist_ok=True)

    # === GT waternark ===
    print("Watermarker: ", args.watermarker)
    watermarked_file = os.path.join(".", "dataset", args.watermarker, args.dataset, "water_mark.csv")
    watermarked_data = pd.read_csv(watermarked_file)
    watermark_gt_str = watermarked_data.iloc[0]["Encoder"]
    if watermark_gt_str[0] == "[":  # Some historical none distructive bug :( will cause this reformatting
        watermark_gt_str = eval(watermark_gt_str)[0]
    watermark_gt = watermark_str_to_numpy(watermark_gt_str)

    # === Init watermarker ===
    watermarker_configs = {
        "watermarker": args.watermarker,
        "watermark_gt": watermark_gt
    }
    watermarker = get_watermarkers(watermarker_configs)
    # Init the dict to save watermarking summary
    save_csv_dir = os.path.join(output_root_path, "water_mark.csv")
    res_dict = {
        "ImageName": [],
        "Decoder": [],
    }


if __name__ == "__main__":
    print("Use this script to download DiffusionDB dataset.")
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--random_seed", dest="random_seed", type=int, help="Manually set random seed for reproduction.",
        default=13
    )
    parser.add_argument(
        '--clean_data_root', type=str, help="Root dir where the clean image dataset is located.",
        default=os.path.join(".", "dataset", "Clean")
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, help="The dataset name: [COCO, DiffusionDB]",
        default="COCO"
    )
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, help="Specification of watermarking method. ['dwtDctSvd', 'rivaGan']",
        default="dwtDctSvd"
    )
    args = parser.parse_args()
    main(args)
    print("Completd")