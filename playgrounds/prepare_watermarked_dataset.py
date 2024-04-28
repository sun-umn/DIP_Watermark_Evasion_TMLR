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
import pandas as pd
from utils.general import rgb2bgr, save_image_bgr, set_random_seeds, \
    watermark_np_to_str
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
    watermark_str = watermark_np_to_str(watermark_gt)


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

    for img_name in img_files:
        img_clean_path = os.path.join(dataset_input_path, img_name)
        print("***** ***** ***** *****")
        print("Processing Image: {} ...".format(img_clean_path))

        img_w_path = os.path.join(output_img_root, img_name)
        watermarker.encode(img_clean_path, img_w_path)
        print("Watermarked img saved to: {}".format(img_w_path))

        # === Sanity Check if watermark is embedded successfully ===
        watermark_decode = watermarker.decode_from_path(img_w_path)
        bitwise_acc_0 = np.mean(watermark_decode == watermark_gt)
        print("Decode the watermarked image for sanity check (bitwise acc. should be close to 100 %)")
        print("  Bitwise acc. {:02f} %".format(bitwise_acc_0 * 100))
        watermark_decode_str = watermark_np_to_str(watermark_decode)

        res_dict["ImageName"].append(img_name)
        res_dict["Encoder"].append([watermark_str])
        res_dict["Decoder"].append([watermark_decode_str])
        res_dict["Match"].append(bitwise_acc_0 > 0.95)
    
    df = pd.DataFrame(res_dict)
    df.to_csv(save_csv_dir, index=False)
    

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
        "--watermarker", dest="watermarker", type=str, help="Specification of watermarking method. ['dwtDctSvd', 'rivaGan']",
        default="dwtDctSvd"
    )
    args = parser.parse_args()
    main(args)
    print("Completd")