import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)

import argparse, torch, pickle

from utils.data_loader import WatermarkedImageDataset
from utils.general import save_image_bgr
from evations import get_interm_collection_algo


def main(args):
    # === Get watermarked data ===
    dataset_root_dir = os.path.join(".", "dataset", args.watermarker, args.dataset)
    is_stegastamp = args.watermarker == "StegaStamp"
    dataset = WatermarkedImageDataset(dataset_root_dir, is_stegastamp)
    print("Experimenting dataset: {}".format(dataset_root_dir))

    # === Constrcut config wrapper ===
    CONFIGS = {
        "dip": {
            "arch": args.arch,   # Used in DIP to select the variant architecture
            "show_every": 5 if args.arch == "vanila" else 2,
            "total_iters": 500 if args.arch == "vanila" else 150, # Used in DIP as the max_iter

            "device": torch.device("cuda"),
            "dtype": torch.float,
        },

        "vae": {
            "arch": args.arch,   # Used in vae to select the variant architecture
            "device": torch.device("cuda"),
        },

        "corrupters": {
            "arch": args.arch,
        },

        "diffuser": {
            "arch": "dummy",  # No second option for diffusion model
        },

        "diffpure": {
            "arch": args.arch,  # No need for second option for diffusion model
            "is_stegastamp": is_stegastamp
        }
    }
    evade_cfgs = CONFIGS[args.evade_method]
    
    # === Create Path to save exp results ===
    log_root_dir = os.path.join("Result-Interm", args.watermarker, args.dataset, args.evade_method, "{}".format(evade_cfgs["arch"]))
    os.makedirs(log_root_dir, exist_ok=True)

    num_images = len(dataset)
    print("Total num. of images: {}".format(num_images))
    print("Interm. collection started ...")

    for idx in range(num_images):
        sample_data = dataset[idx]

        watermark_gt_str = eval(sample_data["watermark_gt_str"])[0]
        watermark_encoded_str = eval(sample_data["watermark_encoded_str"])[0]
        img_name = sample_data["image_name"]

        if watermark_gt_str == watermark_encoded_str:
            # Interm Result save path    
            save_res_name = os.path.join(log_root_dir, "{}.pkl".format(img_name))

            # === Watermark Evasion process (interm. result registration) ===
            evader = get_interm_collection_algo(args.evade_method, args.arch)
            im_w_bgr_uint8 = sample_data["image_bgr_uint8"]
            interm_res = evader(im_w_bgr_uint8, evade_cfgs)
            
            # Append the gt watermark into the file for easier processing later.
            interm_res["watermark_gt_str"] = watermark_gt_str  

            # === save result to pkl ===
            with open(save_res_name, 'wb') as f:
                pickle.dump(interm_res, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("{} recon. result saved to path: {}".format(img_name, save_res_name))
            print("\n")

        else:
            print("Watermark of {} does not work properly using {} watermarker.".format(img_name, args.watermarker))
            print("Skip recon.  \n")
    
    # # === Test Vis ===
    # test_img = interm_res["interm_recon"][0]
    # save_name = "Vis-test-{}-{}-0.png".format(args.evade_method, args.arch)
    # save_image_bgr(test_img, save_name)

    # test_img = interm_res["interm_recon"][-1]
    # save_name = "Vis-test-{}-{}-1.png".format(args.evade_method, args.arch)
    # save_image_bgr(test_img, save_name)
    


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
        default="StegaStamp"
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, help="Dataset [COCO, DiffusionDB]",
        default="COCO"
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
                diffpure --- 0.1  # Do not need other options for this benchmark
        """,
        default="0.1"
    )
    args = parser.parse_args()
    main(args)
    print("\n***** Completed. *****\n")