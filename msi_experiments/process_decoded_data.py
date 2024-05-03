
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)

import argparse, pickle, os, cv2
import numpy as np


def main(args):
    data_root_dir = os.path.join("Result-Decoded", args.watermarker, args.dataset, args.evade_method, args.arch)
    file_path = os.path.join(data_root_dir, "Img-50.pkl")

    # Load Data
    with open(file_path, 'rb') as handle:
        data_dict = pickle.load(handle)

    a_gt = data_dict["watermark_gt_str"]
    aa = data_dict["watermark_decoded"]
    bb = data_dict["psnr_w"]

    a = aa[-2]
    log = []
    for i in range(len(a_gt)):
        if a_gt[i] == a[i]:
            log.append(1)
        else:
            log.append(0)
    bit_acc = np.mean(log)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, 
        help="Specification of watermarking method. [rivaGan, dwtDctSvd]",
        default="dwtDctSvd"
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
        """,
        default="cheng2020-anchor"
    )
    args = parser.parse_args()
    main(args)
    print("Completed.")