import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import math, torch, cv2, argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt


watermarkers = [
    "dwtDctSvd",
    "rivaGan",
    "SSL",
    "SteganoGAN",
    "StegaStamp"
]


def main(args):
    dataset = args.dataset
    for i in range(1, 100, 1):
        # im_name = args.im_name
        im_name = "Img-%d.png" % i

        im_clean_path = os.path.join("dataset", "Clean", dataset, im_name)
        im_clean_bgr_uint8 = cv2.imread(im_clean_path)
        im_clean_bgr_int = im_clean_bgr_uint8.astype(np.int32)
        
        histo_dict = {}
        quantile_dict = {}
        for watermarker in watermarkers:
            histo_dict[watermarker] = []
            quantile_dict[watermarker] = []

        for watermarker in watermarkers:
            if watermarker == "StegaStamp":
                im_w_name = im_name.replace(".png", "_hidden.png")
            else:
                im_w_name = im_name
            im_w_path = os.path.join("dataset", watermarker, dataset, "encoder_img", im_w_name)
            if os.path.exists(im_w_path):
                im_w_bgr_uint8 = cv2.imread(im_w_path)
                im_w_bgr_int = im_w_bgr_uint8.astype(np.int32)
                
                if watermarker == "StegaStamp":
                    img_clean = cv2.resize(im_clean_bgr_uint8, (400, 400), interpolation=cv2.INTER_AREA).astype(np.int32)
                else:
                    img_clean = im_clean_bgr_int

                err_values = np.abs(im_w_bgr_int - img_clean).flatten()
                max_value = np.amax(err_values)
                counts, bins = np.histogram(err_values, bins=(max_value+1))
                quantile = np.quantile(err_values, 0.9)

                histo_dict[watermarker].append((counts, bins))
                quantile_dict[watermarker].append(quantile)
        # histo_dict[watermarker] = (counts, bins)
        # quantile_dict[watermarker] = quantile
        # print(watermarker, np.amax(bins), quantile)
        # print()

    for watermarker in watermarkers:
        quantiles = quantile_dict[watermarker]
        q_mean, q_std = np.mean(quantiles), np.std(quantiles)
        print(watermarker, q_mean, q_std)


    # === Plot Histogram of Img-1 to visualize the err-pixel distribution ===
    
    

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--dataset", dest="dataset", type=str, help="Dataset name.",
        default="COCO"
    )
    parser.add_argument(
        "--im_name", dest="im_name", type=str, help="Clean image name.",
        default="Img-1.png"
    )
    args = parser.parse_args()

    main(args)

    print(" ****** Completed ****** ")