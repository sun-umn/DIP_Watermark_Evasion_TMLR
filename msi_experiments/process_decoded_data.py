
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)

import argparse, pickle, os, cv2


def main(args):
    data_root_dir = os.path.join("Result-Interm", args.watermarker, args.dataset, args.evade_method, args.arch)
    file_path = os.path.join(data_root_dir, "Img-1_hidden.pkl")

    # Load Data
    with open(file_path, 'rb') as handle:
        data_dict = pickle.load(handle)

    print("Shape Check: {} - {} - {} - {}".format(args.watermarker, args.dataset, args.evade_method, args.arch))
    print(data_dict["interm_recon"][0].shape)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, 
        help="Specification of watermarking method. [rivaGan, dwtDctSvd]",
        default="StegaStamp"
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
        default="bmshj2018-factorized"
    )
    args = parser.parse_args()
    main(args)
    print("Completed.")