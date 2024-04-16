import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import os, argparse
import cv2
import numpy as np
from utils.general import rgb2bgr, save_image_bgr


def main(args):

    imgs_dir = args.coco_dir
    # === Specify save path ===
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)

    img_names = [f for f in os.listdir(imgs_dir) if f.split(".")[-1] in ["jpg", ".jpeg", ".png"]]
    img_id = 0
    for img_name in img_names:
        img_path = os.path.join(imgs_dir, img_name)
        test_img = cv2.imread(img_path)
        print("Img-{} shape: {}".format(img_name, test_img.shape))
        # Center crop by the shorter edge
        center = test_img.shape
        h, w, _ = center
        l = min(h, w)
        x = center[1]/2 - l/2
        y = center[0]/2 - l/2
        crop_img = test_img[int(y):int(y+l), int(x):int(x+l)]

        img_id += 1
        save_name = os.path.join(save_root, "Img-{:d}.png".format(img_id))
        save_img = cv2.resize(
            crop_img, (512, 512), 
            interpolation=cv2.INTER_LINEAR
        )
        save_image_bgr(save_img, save_name)
        if img_id >= args.num_images:
            break
    print("Collected in total [{}] images".format(img_id))


if __name__ == "__main__":
    print("Use this script to download DiffusionDB dataset.")
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        '--coco_dir', type=str, help="Places to read in the pre-downloaded coco images.",
        default="/mnt/c/Users/Liang/dataset/COCO/val2017/val2017"
    )
    parser.add_argument(
        "--save_root", dest="save_root", help="Place to save the processed coco dataset.",
        default=os.path.join(".", "examples", "COCO")
    )
    parser.add_argument(
        "--num_images", dest="num_images", type=int, help="Number of DiffusionDB images to retrieve.",
        default=100
    )
    args = parser.parse_args()
    main(args)
    print("Completd")