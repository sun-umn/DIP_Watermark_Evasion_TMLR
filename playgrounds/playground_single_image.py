import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import math, torch, cv2, argparse
import numpy as np
import pickle

# === Project Import ===
from watermarkers import get_watermarkers
from evations import get_evasion_alg
from utils.plottings import plot_dip_res, plot_vae_res, plot_corruption_res, \
    plot_diffuser_res
from utils.general import watermark_np_to_str, set_random_seeds


def main(args):
    # === Some Dummy Configs ===
    device = torch.device("cuda")
    set_random_seeds(args.random_seed)

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

    # === Initiate a watermark ==> in ndarray
    watermark_gt = np.random.binomial(1, 0.5, 32)  

    # === Initiate a encoder & decoder ===
    watermarker_configs = {
        "watermarker": args.watermarker,
        "watermark_gt": watermark_gt
    }
    watermarker = get_watermarkers(watermarker_configs)
    
    # Generated watermarked image and save it to img_w_path
    watermarker.encode(img_clean_path, img_w_path)
    # Check decoding in case learning-based encoder/decoder doesn't work properly
    watermark_decode = watermarker.decode_from_path(img_w_path)
    bitwise_acc_0 = np.mean(watermark_decode == watermark_gt)
    print("*Sanity check for watermarker encoder & decoder:")
    print("  Decoded watermark from im_w: {}".format(watermark_np_to_str(watermark_decode)))
    print("  Bitwise acc. - [{:.04f} %]".format(bitwise_acc_0 * 100))
    assert bitwise_acc_0 > 0.99, "The encoder & decode fails to work on this watermark string."
    # ##### #### #### #### ######

    # === Get Evasion algorithm ===
    detection_threshold = args.detection_threshold
    print("Setting detection threshold [{:02f}] for the watermark detector.".format(detection_threshold))
    evader = get_evasion_alg(args.evade_method, args.arch)

    # Read configs and execude evasions
    CONFIGS = {
        "dip": {
            "arch": args.arch,   # Change this if you want to explore random projector
            "show_every": 5 if args.arch=="vanila" else 2,   
            "total_iters": 500 if args.arch=="vanila" else 100,        

            "device": device,
            "dtype": torch.float,
            "detection_threshold": detection_threshold,
            "verbose": True,
            "save_interms": True
        },

        "vae": {
            "arch": "cheng2020-anchor",   # Change this to explore other pretrained VAE

            "device": torch.device("cuda"),
            "detection_threshold": detection_threshold,
            "verbose": True,
        },

        "corrupters": {
            "arch": args.arch,   # Specify which common corruption to explore
            "detection_threshold": detection_threshold,
            "verbose": True,
        },

        "diffuser": {
            "arch": "dummy",
            "detection_threshold": detection_threshold,
            "verbose": True,
        },

        "diffpure":
        {
            "arch": 0.4,
            "verbose": True,
        }
    }
    evader_cfgs = CONFIGS[args.evade_method]
    # Create log folder 
    vis_root_dir = os.path.join(
        ".", "Visualization", "{}".format(args.im_name.split(".")[0]), "{}".format(args.watermarker), "{}".format(args.evade_method), "{}".format(evader_cfgs["arch"])
    )
    os.makedirs(vis_root_dir, exist_ok=True)

    evasion_res = evader(
        img_clean_path, img_w_path,  watermarker, watermark_gt, evader_cfgs
    )

    # === Save Result ===
    result_path = os.path.join(vis_root_dir, "result.pkl")
    with open(result_path, 'wb') as f:
        pickle.dump(evasion_res, f, protocol=pickle.HIGHEST_PROTOCOL)

    # === Vis result ===
    if args.evade_method.lower() == "dip":
        print("Best evade iter: {}".format(evasion_res["best_evade_iter"]))
        print("Best evade PSNR: {:.04f}".format(evasion_res["best_evade_psnr"]))
        plot_dip_res(vis_root_dir, evasion_res, detection_threshold, vis_recon=True)
    elif args.evade_method.lower() == "vae":
        print("Best evade quality: {}".format(evasion_res["best_evade_quality"]))
        print("Best evade PSNR   : {:.04f}".format(evasion_res["best_evade_psnr"]))
        plot_vae_res(vis_root_dir, evasion_res, detection_threshold)
    elif args.evade_method.lower() == "corrupters":
        print("Use - {} - post-processing: ".format(evader_cfgs["arch"]))
        plot_corruption_res(vis_root_dir, evasion_res, detection_threshold, method_name=evader_cfgs["arch"])
    elif args.evade_method.lower() in ["diffuser", "diffpure"]:
        plot_diffuser_res(vis_root_dir, evasion_res)
    else:
        raise RuntimeError("Un-implemented result summary")


if __name__ == "__main__": 

    print("\n***** This is demo of single image evasion ***** \n")
    
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--random_seed", dest="random_seed", type=int, help="Manually set random seed for reproduction.",
        default=38
    )
    parser.add_argument(
        '--root_path_im_orig', type=str, help="Root folder to the clean images.",
        default=os.path.join("examples", "ori_imgs")
    )
    parser.add_argument(
        "--im_name", dest="im_name", type=str, help="clean image name.",
        default="Img-1.png"
    )
    parser.add_argument(
        "--root_path_im_w", dest="root_path_im_w", type=str, help="Root folder to save watermarked image.",
        default=os.path.join("examples", "watermarked_imgs")
    )
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, help="Specification of watermarking method.",
        default="rivaGan"
    )
    parser.add_argument(
        "--evade_method", dest="evade_method", type=str, help="Specification of evasion method.",
        default="dip"
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
                diffpure --- [0.1, 0.2, 0.4] representing the step params
        """,
        default="vanila"
    )
    parser.add_argument(
        "--detection_threshold", dest="detection_threshold", type=float, default=0.75,
        help="Tunable threhsold to check if the evasion is successful."
    )
    args = parser.parse_args()
    main(args)

    print("\n***** Completed. *****\n")

