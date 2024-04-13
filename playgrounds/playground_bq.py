"""
    A test script to get WevadeBQ on.
"""
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import math, torch, cv2, argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# === Project Import ===
from utils.general import uint8_to_float, img_np_to_tensor
from watermarkers import get_watermarkers
from evations import get_evasion_alg
from utils.plottings import plot_dip_res, plot_vae_res, plot_corruption_res, \
    plot_diffuser_res


# Rewrite The decider into tensor wise operation
class WMTensorDetector(nn.Module):
    def __init__(self, gt, watermarker, th=0.8) -> None:
        self.th = th
        self.decoder = watermarker
        self.gt = torch.from_numpy(gt[np.newaxis, :])

    def forward(self, input_tensor):
        # === Input tensor needs to be [0, 1] tensor
        input_tensor = torch.clamp(input_tensor, 0, 1)
        input_array = (input_tensor.detach().cpu().numpy() * 255).astype(np.uint8)[0, :, :, :]
        input_array = np.transpose(input_array, [1, 2, 0])
        decoded_message = self.decoder.decode(input_array, 'rivaGan')[np.newaxis, :]
        decoded_message = torch.from_numpy(decoded_message)
        bit_acc = 1 - torch.sum(torch.abs(decoded_message-self.gt), 1)/self.gt.shape[1]
        class_idx = torch.logical_or((bit_acc>self.th), (bit_acc<(1-self.th))).long()
        return F.one_hot(class_idx, num_classes=2)

    def predict(self, input_array):
        input_tensor = torch.from_numpy(input_array).to(dtype=torch.float)
        with torch.no_grad():
            return self.forward(input_tensor).cpu().numpy()



def wevade_bq(
    img_clean_path, img_w_path,  watermarker, watermark_gt, evader_cfgs, vis_root
):
    device = evader_cfgs["device"]
    dtype = evader_cfgs["dtype"]

    img_w_bgr = cv2.imread(img_w_path)
    img_w_bgr_float = uint8_to_float(img_w_bgr)
    img_w_bgr_tensor = img_np_to_tensor(img_w_bgr_float).to(device, dtype=dtype)


def main(args):
    # === Some Dummy Configs ===
    device = torch.device("cuda")
    # === Get image paths                          
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
    print("** Sanity check for watermarker encoder & decoder:")
    print("    Bitwise acc. - [{:.04f} %]".format(bitwise_acc_0 * 100))
    assert bitwise_acc_0 > 0.99, "The encoder & decode fails to work on this watermark string."
    # ##### #### #### #### ######

    # === Get Evasion algorithm ===
    detection_threshold = args.detection_threshold
    print("Setting detection threshold [{:02f}] for the watermark detector.".format(detection_threshold))
    # Create log folder 
    vis_root_dir = os.path.join(
        ".", "Visualization", "{}".format(args.im_name.split(".")[0]), "{}".format(args.watermarker), "{}".format(args.evade_method), "{}".format(evader_cfgs["arch"])
    )
    os.makedirs(vis_root_dir, exist_ok=True)
    # ===
    evader_cfgs = {
        "arch": "dummy",
        "device": device,
        "dtype": torch.float,
        "detection_threshold": detection_threshold,
        "verbose": True,    
    }
    evasion_res = wevade_bq(
        img_clean_path, img_w_path,  watermarker, watermark_gt, evader_cfgs, vis_root_dir
    )



if __name__ == "__main__": 

    print("\n***** This is demo of single image evasion ***** \n")
    
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        '--root_path_im_orig', type=str, help="Root folder to the clean images.",
        default=os.path.join("examples", "ori_imgs")
    )
    parser.add_argument(
        "--im_name", dest="im_name", type=str, help="clean image name.",
        default="711.png"
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
        default="Wevadebq"
    )
    parser.add_argument(
        "--arch", dest="arch", type=str, help="Secondary specification of evasion method (if there are other choices).",
        default="dummy"
    )
    parser.add_argument(
        "--detection_threshold", dest="detection_threshold", type=float, default=0.75,
        help="Tunable threhsold to check if the evasion is successful."
    )
    args = parser.parse_args()
    main(args)

    print("\n***** Completed. *****\n")