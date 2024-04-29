import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)

import argparse

from utils.data_loader import WatermarkedImageDataset


def main(args):
    # === Get watermarked data ===
    dataset_root_dir = os.path.join(".", "dataset", args.watermarker, args.dataset)
    dataset = WatermarkedImageDataset(dataset_root_dir)
    print("Experimenting dataset: {}".format(dataset_root_dir))

    # === Create Path to save exp results ===
    log_root_dir = os.path.join("Interm-Result", args.watermarker, args.dataset)
    os.makedirs(log_root_dir, exist_ok=True)

    num_images = len(dataset)
    print("Total num. of images: {}".format(num_images))
    
    for idx in range(num_images):
        sample_data = dataset[idx]

        # Init a dictionary to save all necessary data
        res_dict = {}
        # Interm Result save path
        img_name = sample_data["image_name"]
        save_res_name = os.path.join(log_root_dir, "{}.pkl".format(img_name))

        # === Watermark Evasion process (interm. result registration) ===


        # === save result to pkl ===
        


if __name__ == "__main__": 
    """

        This is the main experiment (interm. result collection) for massive scale experiments.

    """ 
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--random_seed", dest="random_seed", type=int, help="Manually set random seed for reproduction.",
        default=13
    )
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, 
        help="Specification of watermarking method. [rivaGan, dwtDctSvd, SSL, SteganoGAN, StegaStamp]",
        default="rivaGan"
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, help="Dataset [COCO, DiffusionDB]",
        default="COCO"
    )
    parser.add_argument(
        "--evade_method", dest="evade_method", type=str, help="Specification of evasion method.",
        default="dip"
    )
    parser.add_argument(
        "--arch", dest="arch", type=str, help="Secondary specification of evasion method (if there are other choices).",
        default="bm3d"
    )
    args = parser.parse_args()
    main(args)
    print("\n***** Completed. *****\n")