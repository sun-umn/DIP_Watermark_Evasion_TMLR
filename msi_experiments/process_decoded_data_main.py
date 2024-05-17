
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)

import argparse, pickle, os, cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from utils.general import compute_ssim, save_image_bgr
import pandas as pd


def calc_bitwise_acc(gt_str, decoded_str):
    correct, total = 0., 0.
    for i in range(min(len(gt_str), len(decoded_str))):
        if gt_str[i] == decoded_str[i]:
            correct = correct + 1.
        total = total + 1.
    if len(gt_str) != len(decoded_str):
        total += abs(len(gt_str) - len(decoded_str))
    return correct / total


def main(args):
    data_root_dir = os.path.join("Result-Decoded", args.watermarker, args.dataset, args.evade_method, args.arch)
    file_names = [f for f in os.listdir(data_root_dir) if ".pkl" in f]

    # path to get im_clean and im_w
    im_w_root_dir = os.path.join("dataset", args.watermarker, args.dataset, "encoder_img")
    im_orig_root_dir = os.path.join("dataset", "Clean", args.dataset)

    # path to get the interm result (for ssim, quantile metric calculation)
    interm_root_dir = os.path.join("Result-Interm", args.watermarker, args.dataset, args.evade_method, args.arch)

    # === Create Save directory ===
    save_root_dir = os.path.join("Result-Stats-Summary", args.watermarker, args.dataset, args.evade_method)
    os.makedirs(save_root_dir, exist_ok=True)
    save_file_name = os.path.join(save_root_dir, "{}.pkl".format(args.arch))

    # Logs to save after processing
    psnr_w_to_orig_log = []
    best_psnr_w_log = []
    best_psnr_orig_log = []
    best_ssim_orig_log = []
    best_quantile_log = []
    evade_success_log = []  # Check if this is close to 100%, otherwise discard the shitty evasion algo.
    
    # Collect all interm. results and calc. necessary metrics
    for file_name in file_names:
        file_path = os.path.join(data_root_dir, file_name)

        # (1) Retrieve the im_w name
        im_orig_name = file_name.replace(".pkl", ".png")
        if args.watermarker == "StegaStamp":
            im_w_file_name = im_orig_name.replace(".png", "_hidden.png")
        else:
            im_w_file_name = im_orig_name

        # Readin the im_w into bgr uint8 format
        im_w_path = os.path.join(im_w_root_dir, im_w_file_name)
        im_w_bgr_uint8 = cv2.imread(im_w_path)
        im_w_int = im_w_bgr_uint8.astype(np.int32)

        # Readin im_orig and im_w and calculate this psnr 
        im_orig_path = os.path.join(im_orig_root_dir, im_orig_name)
        im_orig_bgr_uint8 = cv2.imread(im_orig_path)
        if args.watermarker == "StegaStamp":
            im_orig_bgr_uint8 = cv2.resize(im_orig_bgr_uint8, (400, 400), interpolation=cv2.INTER_AREA)
        im_orig_int = im_orig_bgr_uint8.astype(np.int32)

        psnr_w_to_orig = compute_psnr(
            im_orig_int, im_w_int, data_range=255
        )
        # save (1)
        psnr_w_to_orig_log.append(psnr_w_to_orig)

        # Load Data
        with open(file_path, 'rb') as handle:
            data_dict = pickle.load(handle)
        
        index_log = data_dict["index"]
        watermark_gt_str = data_dict["watermark_gt_str"]
        watermark_decoded_str = data_dict["watermark_decoded"]
        psnr_orig_log = data_dict["psnr_orig"]
        psnr_w_log = data_dict["psnr_w"]

        num_interm_data = len(psnr_w_log)

        # === Calc bitwise acc ===
        bitwise_acc_log = []
        for i in range(num_interm_data):
            watermark_decoded = str(watermark_decoded_str[i])
            bitwise_acc = calc_bitwise_acc(watermark_gt_str, watermark_decoded)
            bitwise_acc_log.append(bitwise_acc)

        # === To finde the best evade Iter ===
        best_index, best_psnr_w, best_psnr_orig = None, -float("inf"), None
        detection_threshold = 0.75
        # Exausive search 
        for idx in range(num_interm_data): 
            psnr_w = psnr_w_log[idx]
            psnr_orig = psnr_orig_log[idx]
            bitwise_acc = bitwise_acc_log[idx]

            condition_1 = bitwise_acc < detection_threshold
            condition_2 = psnr_w > best_psnr_w
            if condition_1 and condition_2:
                best_index = idx
                best_psnr_w = psnr_w
                best_psnr_orig = psnr_orig

        if best_index is None:
            evade_success_log.append(0)
        else:
            best_psnr_w_log.append(best_psnr_w)
            best_psnr_orig_log.append(best_psnr_orig)
            evade_success_log.append(1)

            # After getting the best index, calculate the ssim and quantile metric respectively
            if args.watermarker != "StegaStamp": # Retrive the interm file
                interm_file_path = os.path.join(interm_root_dir, file_name)
            else:
                interm_file_path = os.path.join(interm_root_dir, file_name.replace(".pkl", "_hidden.pkl")) 
            # Load Data
            with open(interm_file_path, 'rb') as handle:
                interm_data_dict = pickle.load(handle)
            if args.evade_method == "WevadeBQ":
                img_recon_list = interm_data_dict["best_recon"]
            else:
                img_recon_list = interm_data_dict["interm_recon"]  # A list of recon. image in "bgr uint8 np" format (cv2 standard format)
            best_recon = img_recon_list[best_index]  # bgr_uint8
            if args.arch == "cheng2020-anchor":
                best_recon = cv2.resize(best_recon, (400, 400), interpolation=cv2.INTER_AREA)
            best_recon_int = best_recon.astype(np.int32)

            # SSIM
            ssim_recon_orig = compute_ssim(
                im_orig_int, best_recon_int, data_range=255
            )
            best_ssim_orig_log.append(ssim_recon_orig)

            # Quantile 90 % value
            err_values = np.abs(im_orig_int - best_recon_int).flatten()
            quantile = np.quantile(err_values, 0.9)
            best_quantile_log.append(quantile)

            # === Sanity Check === 1) Vis im_orig; 2) Vis im_best_recon; 3) histo of err_values
            if file_names[0] == file_name:
                save_name = os.path.join(save_root_dir, "{}_{}_orig.png".format(args.evade_method, args.arch))
                save_image_bgr(im_orig_bgr_uint8, save_name)
                save_name = os.path.join(save_root_dir, "{}_{}_recon.png".format(args.evade_method, args.arch))
                save_image_bgr(best_recon, save_name)

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
                n, bins, _ = ax.hist(err_values, bins=np.amax(err_values), alpha=0.6)
                ax.vlines(quantile, ymin=0.01, ymax=1e6, color="black", ls="dashed", lw=2, label="Quantile 0.9")
                ax.set_yscale("log")
                ax.yaxis.grid(True)
                ax.xaxis.grid(False)
                ax.set_xticks([255])
                ax.set_xlim([0, 255])
                ax.set_ylim([0.1, 1e6])
                ax.set_yticks([1e2, 1e5])
                ax.tick_params(axis='y', labelsize=25)
                ax.tick_params(axis='x', labelsize=25)
                # ax.legend(loc='upper right', ncol=1, fancybox=True, shadow=False, fontsize=25, framealpha=0.3)
                plt.tight_layout()
                save_name = os.path.join(save_root_dir, "{}_{}_histo.png".format(args.evade_method, args.arch))
                plt.savefig(save_name)
                plt.close(fig)

        # # === Sanity Check === Plot the psnr and bitwise acc curve
        if file_names[0] == file_name and args.evade_method != "WevadeBQ" and best_index is not None:
            best_index = index_log[best_index]
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            ax[0].plot(index_log, psnr_orig_log, label="PSNR (recon - clean)", color="orange")
            ax[0].plot(index_log, psnr_w_log, label="PSNR (recon - watermarked)", color="blue", ls="dashed")
            if best_index is not None:
                ax[0].vlines(best_index, ymin=np.amin(psnr_w_log), ymax=np.amax(psnr_w_log), color="black", ls="dashed", label="Best Evade Iter")
            ax[0].legend()

            ax[1].plot(index_log, bitwise_acc_log, label="Bitwise Acc.")
            ax[1].hlines(y=detection_threshold, xmin=np.amin(index_log), xmax=np.amax(index_log), ls="dashed", color="black")
            ax[1].hlines(y=(1-detection_threshold), xmin=np.amin(index_log), xmax=np.amax(index_log), ls="dashed", color="black")
            if best_index is not None:
                ax[1].vlines(best_index, ymin=0, ymax=1, color="black", ls="dashed", label="Best Recon Iter")
            ax[1].legend()
            plt.tight_layout()
            save_name = os.path.join(save_root_dir, "{}_{}_psnr_bt_acc.png".format(args.evade_method, args.arch))
            plt.savefig(save_name)

    # === Summarize Data ===
    mean_psnr_w, std_psnr_w = np.mean(best_psnr_w_log), np.std(best_psnr_w_log)
    mean_psnr_orig, std_psnr_orig = np.mean(best_psnr_orig_log), np.std(best_psnr_orig_log)
    mean_ssim_orig, std_ssim_orig = np.mean(best_ssim_orig_log), np.std(best_ssim_orig_log)
    mean_psnr_w_to_orig, std_psnr_w_to_orig = np.mean(psnr_w_to_orig_log), np.std(psnr_w_to_orig_log)
    mean_quantile, std_quantile = np.mean(best_quantile_log), np.std(best_quantile_log)
    evade_success_rate = np.mean(evade_success_log)
    print("===== Processed Summary: Watermarker [{}] - Dataset [{}] =====".format(args.watermarker, args.dataset))
    print("PSNR w to orig: Mean {:.02f} - std({:.02f})".format(mean_psnr_w_to_orig, std_psnr_w_to_orig))
    print("Best PSNR-Orig: Mean - STD: ")
    print("  [{:.02f}] - [{:.02f}]".format(mean_psnr_orig, std_psnr_orig))
    print("Best SSIM-Orig: Mean - STD: ")
    print("  [{:.02f}] - [{:.02f}]".format(mean_ssim_orig, std_ssim_orig))
    print("Best Quantile (90%) watermark pixel value: ")
    print("  [{:.02f}] - [{:.02f}]".format(mean_quantile, std_quantile))
    # print("Best PSNR-W: Mean - STD: ")
    # print("  [{:.02f}] - [{:.02f}]".format(mean_psnr_w, std_psnr_w))
    print("Evasion success rate: {:.03f} %".format(evade_success_rate * 100))

    # === Save processed data ===
    save_dict = {
        "psnr_w_to_orig": psnr_w_to_orig_log,
        "best_psnr_w": best_psnr_w_log,
        "best_psnr_orig": best_psnr_orig_log,
        "best_ssim_orig": best_ssim_orig_log,
        "best_quantile_log": best_quantile_log,
        "evade_success_rate": [np.mean(evade_success_log)]
    }
    # with open(save_file_name, 'wb') as handle:
    #     pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print("Decoded Interm. result saved to {}".format(save_file_name))

    save_file_name = save_file_name.replace(".pkl", ".csv")
    df = pd.DataFrame.from_dict(save_dict, orient='index')
    df.to_csv(save_file_name)
    print("Decoded Interm. result saved to {}".format(save_file_name))
    print("=========================================================== \n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, 
        help="Specification of watermarking method. [rivaGan, dwtDctSvd]",
        default="dwtDctSvd"
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, 
        help="Dataset [COCO, DiffusionDB]",
        default="COCO"
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
                diffpure --- ["0.1", "0.2", "0.3"]
        """,
        default="vanila"
    )
    args = parser.parse_args()
    # main(args)

    root_lv1 = os.path.join("Result-Decoded", args.watermarker, args.dataset)
    corrupter_names = [f for f in os.listdir(root_lv1)]
    for corrupter in corrupter_names:
        root_lv2 = os.path.join(root_lv1, corrupter)
        arch_names = [f for f in os.listdir(root_lv2) if "blur" not in f]
        for arch in arch_names:
            args.evade_method = corrupter
            args.arch = arch
            print("Processing: {} - {} - {} - {}".format(args.watermarker, args.dataset, args.evade_method, args.arch))
            main(args)
    print("Completed.")