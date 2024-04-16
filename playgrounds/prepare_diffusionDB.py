import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import os, argparse
from datasets import load_dataset
import numpy as np
from utils.general import rgb2bgr, save_image_bgr
"""
    Use this script to download and prepare the DiffusionDB images (which has been pulled in examples/DiffusionDB with this repo.)
"""

def main(args):

    dataset = load_dataset('poloclub/diffusiondb', 'large_random_1k')["train"]
    folder_name = "DiffusionDB"
    # === Specify save path ===
    root_dir = os.path.join(args.root_dir, folder_name)
    os.makedirs(root_dir, exist_ok=True)

    img_id = 0
    for sample_idx in range(len(dataset)):
        test_sample = dataset[sample_idx]
        test_img = rgb2bgr(np.array(test_sample["image"]))
        print("Img-{} shape: {}".format(sample_idx, test_img.shape))
        if test_img.shape[0] == test_img.shape[1] == 512:
            img_id += 1
            save_name = os.path.join(root_dir, "Img-{:d}.png".format(img_id))
            save_image_bgr(test_img, save_name)
            if img_id >= args.num_images:
                break
    print("Collected in total [{}] images".format(img_id))


if __name__ == "__main__":
    print("Use this script to download DiffusionDB dataset.")
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        '--root_dir', type=str, help="Root folder to the clean images.",
        default=os.path.join(".", "examples")
    )
    parser.add_argument(
        "--num_images", dest="num_images", type=int, help="Number of DiffusionDB images to retrieve.",
        default=100
    )
    args = parser.parse_args()
    main(args)
    print("Completd")