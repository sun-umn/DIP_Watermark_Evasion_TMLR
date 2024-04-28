"""
    This scripts is used to create a dataset composed of:
        1) watermarked images
        2) a summary CSV file
    using:
        1) a dataset of clean images
        2) a randomly generated watermark of 32 bits.

    This script provides the option of 1) rivaGan 2) dctDwtSVD;
    Other invisible watermarkers (encoder and decoder) are not provided in this script and we recomment the readers to use this as the template
    and refer to their original github repo, respectively.
"""
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import os, argparse, torch
import cv2
import numpy as np
from utils.general import rgb2bgr, save_image_bgr, set_random_seeds
from watermarkers import get_watermarkers


def main(args):
    # === Some dummt configs ===
    device = torch.device("cuda")
    set_random_seeds(args.random_seed)

    # === Set the dataset paths ===
    dataset_input_path = os.path.join(
        args.clean_data_root, args.dataset_name
    )
    output_root_path = os.path.join(
        ".", "dataset", args.watermarker, args.dataset_name
    )
    output_img_root = os.path.join(output_root_path, "encoder_img")
    os.makedirs(output_img_root, exist_ok=True)

    # === Scan all images in the dataset (clean) ===
    img_files = [f for f in os.listdir(dataset_input_path) if ".png" in f]
    print("Total number of images: [{}] --- (Should be 100)".format(len(img_files)))

    # === Init a random watermark ===
    watermark_gt = np.random.binomial(1, 0.5, 32) 
    # watermark_gt = np.zeros(32)  # This will fail the rivaGan, same for np.ones(32) 
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
        "Encoder": [],
        "Decoder": [],
        "Match": []
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
        "--dataset_name", dest="dataset_name", type=str, help="The dataset name: [COCO, DiffusionDB]",
        default="COCO"
    )
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, help="Specification of watermarking method.",
        default="rivaGan"
    )
    args = parser.parse_args()
    main(args)
    print("Completd")