import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import math, torch, cv2, argparse
import numpy as np

# === Project Import ===
from utils.general import watermark_np_to_str
from utils.build import get_watermarkers


def main(args):
    device = torch.device("cuda")   

    example_img_path = args.example_img_path
    # === Read in image ==> 1) bgr 2) uint8
    img_orig_bgr = cv2.imread(example_img_path)

    # === Initiate a watermark ==> in ndarray
    watermark_gt = np.random.binomial(1, 0.5, 32)  
    watermark_str = watermark_np_to_str(watermark_gt)

    # === Initiate a encoder & decoder ===
    watermarker_configs = {
        "watermarker": args.watermarker,
        "watermark_gt": watermark_gt
    }
    watermarker = get_watermarkers(watermarker_configs)
    print("")


if __name__ == "__main__":
    print("\n***** This is a single image demo to evade invisible watermark by DIP ***** \n")
    
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        '--example_img_path', type=str, help="Path to the single image example",
        default=os.path.join("examples", "ori_imgs", "000000000711.png")
    )
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, help="Speficifation of watermarking method.",
        default="dwtDctSvd"
    )
    args = parser.parse_args()
    main(args)

    print("\n***** Completed. *****\n")