import argparse
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
from utils.data_loader import WatermarkedImageDataset


def main(args):
    # === Get watermarked data ===
    dataset_root_dir = os.path.join(".", "dataset", args.watermarker, args.dataset)
    is_stegastamp = args.watermarker == "StegaStamp"
    is_tree_ring = args.watermarker == "Tree-Ring"
    dataset = WatermarkedImageDataset(dataset_root_dir, is_stegastamp, is_tree_ring)
    print("Experimenting dataset: {}".format(dataset_root_dir))
    print("  {} | {} ".format(args.evade_method, args.arch))

    # === Get result folder ===
    result_dir = os.path.join(
        args.res_type, args.watermarker, args.dataset, args.evade_method, args.arch
    )
    result_file_list = [f for f in os.listdir(result_dir)]

    for idx in range(len(dataset)):
        sample_data = dataset[idx]
        img_name = sample_data["image_name"]
        file_str = img_name.split(".")[0] + ".pkl"

        if file_str in result_file_list:
            pass
        else:
            print("Data unprocessed - Index [{}] | File [{}]".format(idx, img_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, 
        help="Specification of watermarking method. [rivaGan, dwtDctSvd, SSL, SteganoGAN, StegaStamp]",
        default="Tree-Ring"
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, help="Dataset [COCO, DiffusionDB]",
        default="Gustavosta"
    )
    parser.add_argument(
        "--evade_method", dest="evade_method", type=str, help="Specification of evasion method.",
        default="diffpure"
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
                diffpure --- dummy
        """,
        default="dummy"
    )
    parser.add_argument(
        "--res_type", dest="res_type", type=str, default="Result-Interm",
        help="Result-Interm | XXX | XXX"
    )
    args = parser.parse_args()
    main(args)
    print("\n***** Completed. *****\n")