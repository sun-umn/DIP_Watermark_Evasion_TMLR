"""
    This script is a skeleton file for **Taihui** to:

    1) Read in the watermark evasion interm. results

    2) Decode each of the interm. result using the encoder/decoder API

    3) Save the result with standardized format
"""
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)

import argparse
import pickle, os
from watermarkers import get_watermarkers
from utils.general import watermark_str_to_numpy, watermark_np_to_str

def main(args):
    # === This is where the interm. results are saved ===
    data_root_dir = os.path.join("Result-Interm", args.watermarker, args.dataset, args.evade_method, args.arch)
    file_names = [f for f in os.listdir(data_root_dir) if ".pkl" in f]  # Data are saved as dictionary in pkl format.

    # === Save the result in a different location in case something went wrong ===
    save_root_dir = os.path.join("Result-Decoded", args.watermarker, args.dataset, args.evade_method, args.arch)
    os.makedirs(save_root_dir)
    
    # === Process each file ===
    for file_name in file_names:
        data_file_path = os.path.join(data_root_dir, file_name)
        with open(data_file_path, 'rb') as handle:
            data_dict = pickle.load(handle)
        
        img_recon_list = data_dict["interm_recon"]  # A list of recon. image in "bgr uint8 np" format (cv2 standard format)
        n_recon = len(img_recon_list)
        print("Total number of interm. recon. to process: [{}]".format(n_recon))

        # === Initiate a encoder & decoder ===
        watermark_gt_str = data_dict["watermark_gt_str"]
        if watermark_gt_str[0] == "[":  # Some historical none distructive bug :( will cause this reformatting
            watermark_gt_str = eval(data_dict["watermark_gt_str"])[0]
        watermark_gt = watermark_str_to_numpy(watermark_gt_str)
        watermarker_configs = {
            "watermarker": args.watermarker,
            "watermark_gt": watermark_gt
        }
        watermarker = get_watermarkers(watermarker_configs)

        # Process each inter. recon
        watermark_decoded_log = []  # A list to save decoded watermark
        for img_idx in range(n_recon):
            img_bgr_uint8 = img_recon_list[img_idx]    # shape [512, 512, 3]
            
            # =================== YOUR CODE HERE =========================== #
            
            # Step 0: if you need to change the input format
            img_input = img_bgr_uint8

            # Step 1: Decode the interm. result
            watermark_decoded = watermarker.decode(img_input)
            watermark_decoded_str = watermark_np_to_str(watermark_decoded)

            # Step 2: log the result
            watermark_decoded_log.append(watermark_decoded_str)

            # ============================================================= #
        
        # Save the result
        data_dict["watermark_decoded"] = watermark_decoded_log
        data_dict["watermark_gt_str"] = watermark_gt_str # Some historical none distructive bug :( will cause this reformatting

        save_name = os.path.join(save_root_dir, file_name)
        with open(save_name, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Decoded Interm. result saved to {}".format(save_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, 
        help="Specification of watermarking method. [rivaGan, dwtDctSvd]",
        default="rivaGan"
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
    print("\n***** Completed. *****\n")