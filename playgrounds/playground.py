import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import math, torch, cv2, argparse
import numpy as np

# === Project Import ===
from utils.general import watermark_np_to_str
from watermarkers import get_watermarkers
from evations import get_evasion_alg
from utils.plottings import plot_dip_res


def main(args):
    # === Some Dummy Configs ===
    device = torch.device("cuda")

    img_clean_path = os.path.join(
        args.root_path_im_orig, args.im_name  # Path to a clean image
    )

    img_w_root_dir = os.path.join(
        args.root_path_im_w, args. watermarker
    )
    os.makedirs(img_w_root_dir, exist_ok=True)
    img_w_path = os.path.join(
        img_w_root_dir, args.im_name  # Path to save the watermarked image.
    )

    vis_root_dir = os.path.join(
        ".", "Visualizations", "{}_{}".format(args.watermarker, args.evade_method)
    )
    os.makedirs(vis_root_dir, exist_ok=True)

    # === Initiate a watermark ==> in ndarray
    watermark_gt = np.random.binomial(1, 0.5, 32)  
    watermark_str = watermark_np_to_str(watermark_gt)

    # === Initiate a encoder & decoder ===
    watermarker_configs = {
        "watermarker": args.watermarker,
        "watermark_gt": watermark_gt
    }
    watermarker = get_watermarkers(watermarker_configs)
    
    # Generated watermarked image and save it to img_w_path
    watermarker.encode(img_clean_path, img_w_path)

    # === Get Evasion algorithm ===
    evader = get_evasion_alg(args.evade_method)
    evader_cfgs = {
        "arch": "vanila",   # Used in DIP to select the variant architecture
        "show_every": 10,   # Used in DIP to log interm. result
        "total_iters": 100, # Used in DIP as the max_iter
        "lr": 0.01,         # Used in DIP as the learning rate
    }
    evasion_res = evader(
        img_clean_path, img_w_path,  watermarker, watermark_gt, evader_cfgs,
        save_interm=True, verbose=True
    )

    print("Best evade iter: {}".format(evasion_res["best_evade_iter"]))
    print("Best evade PSNR: {:.04f}".format(evasion_res["best_evade_psnr"]))
    plot_dip_res(vis_root_dir, evasion_res)


if __name__ == "__main__":
    print("\n***** This is demo of single image evasion ***** \n")
    
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        '--root_path_im_orig', type=str, help="Root folder to the clean images.",
        default=os.path.join("examples", "ori_imgs")
    )
    parser.add_argument(
        "--im_name", dest="im_name", type=str, help="clean image name.",
        default="000000000711.png"
    )
    parser.add_argument(
        "--root_path_im_w", dest="root_path_im_w", type=str, help="Root folder to save watermarked image.",
        default=os.path.join("examples", "watermarked_imgs")
    )

    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, help="Specification of watermarking method.",
        default="dwtDctSvd"
    )
    parser.add_argument(
        "--evade_method", dest="evade_method", type=str, help="Specification of evasion method.",
        default="dip"
    )
    args = parser.parse_args()
    main(args)

    print("\n***** Completed. *****\n")