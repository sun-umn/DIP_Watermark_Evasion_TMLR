import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import math, torch, cv2, argparse
from pytorch_msssim import ssim, ms_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


def compute_ssim(a, b, data_range):
    a = np.transpose(a, [2, 0, 1])
    a = torch.from_numpy(a).to(dtype=torch.float).unsqueeze(0)
    b = np.transpose(b, [2, 0, 1])
    b = torch.from_numpy(b).to(dtype=torch.float).unsqueeze(0)
    return ssim(a, b, data_range=data_range).item()


watermarkers = [
    "dwtDctSvd",
    "rivaGan",
    "Rosteals",
    "SSL",
    "StegaStamp",
    "TrustMark"
]


def main(args):
    dataset = args.dataset
    histo_dict = {}
    quantile_dict = {}
    ssim_dict = {}
    psnr_dict = {}
    for watermarker in watermarkers:
        histo_dict[watermarker] = []
        quantile_dict[watermarker] = []
        ssim_dict[watermarker] = []
        psnr_dict[watermarker] = []

    max_value = 0 
    for i in range(1, 2001, 1):
        # im_name = args.im_name
        im_name = "Img-%d.png" % i

        im_clean_path = os.path.join("dataset", "Clean", dataset, im_name)
        im_clean_bgr_uint8 = cv2.imread(im_clean_path)
        im_clean_bgr_int = im_clean_bgr_uint8.astype(np.int32)
        
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

                # === Compute PSNR ===
                psnr_orig_w = compute_psnr(
                    img_clean, im_w_bgr_int, data_range=255  # PSNR of recon v.s. watermarked img
                )   
                psnr_dict[watermarker].append(psnr_orig_w)
                # === Compute SSIM ===
                ssim_orig_w = compute_ssim(img_clean, im_w_bgr_int, data_range=255)
                ssim_dict[watermarker].append(ssim_orig_w)

                # === Compute quantile measure ===
                err_values = np.abs(im_w_bgr_int - img_clean).flatten()
                max_value_0 = np.amax(err_values)
                counts, bins = np.histogram(err_values, bins=(max_value_0+1))
                max_value = max(max_value_0, max_value)

                quantile = np.quantile(err_values, 0.9)

                histo_dict[watermarker].append((counts, bins))
                quantile_dict[watermarker].append(quantile)


    
    for watermarker in watermarkers:
        print("Watermark {} quality measures: ".format(watermarker))
        quantiles = quantile_dict[watermarker]
        # print("{} - len data {}".format(watermarker, len(quantiles)))
        q_mean, q_std = np.mean(quantiles), np.std(quantiles)
        print("  Quantile (90 %) measure: Mean [{:.02f}] - std [{:.02f}]".format(q_mean, q_std))

        psnrs = psnr_dict[watermarker]
        psnr_mean, psnr_std = np.mean(psnrs), np.std(psnrs)
        print("  PSNR: Mean [{:.02f}] - std [{:.02f}]".format(psnr_mean, psnr_std))

        ssims = ssim_dict[watermarker]
        ssim_mean, ssim_std = np.mean(ssims), np.std(ssims)
        print("  SSIM: Mean [{:.02f}] - std [{:.02f}]".format(ssim_mean, ssim_std))

    # === Vis: Plot histograms (of Img-1) to visualize the err-pixel distribution of different watermark methods ===
    num_figs = len(watermarkers)
    fig, ax = plt.subplots(nrows=num_figs, ncols=1, figsize=(5, 12), sharey=True)
    for idx, watermarker in enumerate(watermarkers):
        ax_idx = idx
        counts, bins = histo_dict[watermarker][0]
        quantile = quantile_dict[watermarker][0]
        ax[ax_idx].hist(bins[:-1], bins, weights=counts, alpha=0.6)
        max_value = 100
        ax[ax_idx].set_xlim([0, max_value])
        ax[ax_idx].set_yscale("log")
        # l1 = ax[ax_idx].vlines(x=quantile, ymin=0.1, ymax=1e6, lw=2, ls="dashed", color="black", label=r"$90 \% ~ quantile ~ (x = {:d})$".format(int(quantile)))
        l1 = ax[ax_idx].vlines(x=quantile, ymin=0.1, ymax=1e6, lw=2, ls="dashed", color="black", label=r"$x = {:d}$".format(int(quantile)))
        ax[ax_idx].yaxis.grid(True)
        ax[ax_idx].xaxis.grid(False)
        ax[ax_idx].set_yticks([1e2, 1e5])
        ax[ax_idx].tick_params(axis='y', labelsize=15)
        ax[ax_idx].set_xticks([max_value])
        ax[ax_idx].tick_params(axis='x', labelsize=15)
        ax[ax_idx].set_title(watermarker)
        ax[ax_idx].legend(
            loc='upper right',
            # bbox_to_anchor=(0.7, 1),
            ncol=1, fancybox=True, shadow=False, fontsize=15, framealpha=0.3
        )
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--dataset", dest="dataset", type=str, help="Dataset name.",
        default="DiffusionDB"
    )
    parser.add_argument(
        "--im_name", dest="im_name", type=str, help="Clean image name.",
        default="Img-1.png"
    )
    args = parser.parse_args()

    main(args)

    print(" ****** Completed ****** ")