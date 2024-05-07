
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


def calc_bitwise_acc(gt_str, decoded_str):
    correct, total = 0., 0.
    for i in range(len(gt_str)):
        if gt_str[i] == decoded_str[i]:
            correct = correct + 1.
        total = total + 1.
    return correct / total


def main(args):
    data_root_dir = os.path.join("Result-Decoded", args.watermarker, args.dataset, args.evade_method, args.arch)
    file_names = [f for f in os.listdir(data_root_dir) if ".pkl" in f]

    # path to get im_clean and im_w
    im_w_root_dir = os.path.join("dataset", args.watermarker, args.dataset, "encoder_img")
    im_orig_root_dir = os.path.join("dataset", "Clean", args.dataset)

    # === Create Save directory ===
    save_root_dir = os.path.join("Result-Stats-Summary", args.watermarker, args.dataset, args.evade_method)
    os.makedirs(save_root_dir, exist_ok=True)
    save_file_name = os.path.join(save_root_dir, "{}.pkl".format(args.arch))

    # Logs to save after processing
    psnr_w_to_orig_log = []
    best_psnr_w_log = []
    best_psnr_orig_log = []
    evade_success_log = []
    
    # Collect all interm. results and calc. necessary metrics
    for file_name in file_names:
        file_path = os.path.join(data_root_dir, file_name)

        

        # (1) Retrieve the im_w name
        im_w_file_name = file_name.replace(".pkl", ".png")
        if "_hidden" in im_w_file_name:
            im_orig_name = im_w_file_name.replace("_hidden", "")
        else:
            im_orig_name = im_w_file_name
        # Readin the im_w into bgr uint8 format
        im_w_path = os.path.join(im_w_root_dir, im_w_file_name)
        im_w_bgr_uint8 = cv2.imread(im_w_path)
        # Readin im_orig and im_w and calculate this psnr 
        im_orig_path = os.path.join(im_orig_root_dir, im_orig_name)
        im_orig_bgr_uint8 = cv2.imread(im_orig_path)
        psnr_w_to_orig = compute_psnr(
            im_orig_bgr_uint8.astype(np.int16), im_w_bgr_uint8.astype(np.int16), data_range=255
        )
        # save (1)
        psnr_w_to_orig_log.append(psnr_w_to_orig)

        # Load Data
        with open(file_path, 'rb') as handle:
            data_dict = pickle.load(handle)
        
        index_log = data_dict["index"]
        watermark_gt_str = data_dict["watermark_gt_str"]
        watermark_decoded_str = data_dict["watermark_decoded"]
        mse_orig_log = data_dict["mse_orig"]
        mse_w_log = data_dict["mse_w"]
        psnr_orig_log = data_dict["psnr_orig"]
        psnr_w_log = data_dict["psnr_w"]

        num_interm_data = len(psnr_w_log)
        print("Total number of interm data: {}".format(num_interm_data))

        # === Calc bitwise acc ===
        bitwise_acc_log = []
        for i in range(num_interm_data):
            bitwise_acc = calc_bitwise_acc(watermark_gt_str, watermark_decoded_str[i])
            bitwise_acc_log.append(bitwise_acc)

        # === To finde the best evade Iter ===
        best_index, best_psnr_w, best_psnr_orig = None, -float("inf"), None
        detection_threshold = 0.75
        for idx in range(num_interm_data-1, -1, -1):
            psnr_w = psnr_w_log[idx]
            psnr_orig = psnr_orig_log[idx]
            bitwise_acc = bitwise_acc_log[idx]

            condition_1 = (1-detection_threshold) < bitwise_acc < detection_threshold
            condition_2 = psnr_w > best_psnr_w
            if condition_1 and condition_2:
                best_index = index_log[idx]
                best_psnr_w = psnr_w
                best_psnr_orig = psnr_orig
        print("Best evade iter [{}] with PSNR-W [{}] & PSNR-orig [{}]".format(best_index, best_psnr_w, best_psnr_orig))

        if best_index is None:
            print("Does not evade successfully")
            # best_psnr_w_log = []
            # best_psnr_orig_log = []
            evade_success_log.append(0)
        else:
            best_psnr_w_log.append(best_psnr_w)
            best_psnr_orig_log.append(best_psnr_orig)
            evade_success_log.append(1)

        # # === Sanity Check === Plot the psnr and bitwise acc curve
        # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        # ax[0].plot(index_log, psnr_orig_log, label="PSNR (recon - clean)", color="orange")
        # ax[0].plot(index_log, psnr_w_log, label="PSNR (recon - watermarked)", color="blue", ls="dashed")
        # if best_index is not None:
        #     ax[0].vlines(best_index, ymin=np.amin(psnr_w_log), ymax=np.amax(psnr_w_log), color="black", ls="dashed", label="Best Evade Iter")
        # ax[0].legend()

        # ax[1].plot(index_log, bitwise_acc_log, label="Bitwise Acc.")
        # ax[1].hlines(y=detection_threshold, xmin=np.amin(index_log), xmax=np.amax(index_log), ls="dashed", color="black")
        # ax[1].hlines(y=(1-detection_threshold), xmin=np.amin(index_log), xmax=np.amax(index_log), ls="dashed", color="black")
        # if best_index is not None:
        #     ax[1].vlines(best_index, ymin=0, ymax=1, color="black", ls="dashed", label="Best Recon Iter")
        # ax[1].legend()
        # plt.tight_layout()
        # # save_name = os.path.join(save_root, "psnr_bt_acc.png")
        # # plt.savefig(save_name)
        # plt.show()
        # plt.close(fig)
        
    # === Summarize Data ===
    mean_psnr_w, std_psnr_w = np.mean(best_psnr_w_log), np.std(best_psnr_w_log)
    mean_psnr_orig, std_psnr_orig = np.mean(best_psnr_orig_log), np.std(best_psnr_orig_log)
    mean_psnr_w_to_orig, std_psnr_w_to_orig = np.mean(psnr_w_to_orig_log), np.std(psnr_w_to_orig_log)
    evade_success_rate = np.mean(evade_success_log)
    print("===== Processed Summary: Watermarker [{}] - Dataset [{}] =====".format(args.watermarker, args.dataset))
    print("PSNR ot orig: Mean {:03f} - std({:03f})".format(mean_psnr_w_to_orig, std_psnr_w_to_orig))
    print("Best PSNR-W: Mean - STD: ")
    print("  [{:03f}] - [{:03f}]".format(mean_psnr_w, std_psnr_w))
    print("Best PSNR-Orig: Mean - STD: ")
    print("  [{:03f}] - [{:03f}]".format(mean_psnr_orig, std_psnr_orig))
    print("Evasion success rate: {:.03f} %".format(evade_success_rate * 100))

    # === Save processed data ===
    save_dict = {
        "psnr_w_to_orig": psnr_w_to_orig_log,
        "best_psnr_w": best_psnr_w_log,
        "best_psnr_orig": best_psnr_orig_log,
        "evade_success_rate": np.mean(evade_success_log)
    }
    with open(save_file_name, 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Decoded Interm. result saved to {}".format(save_file_name))
        
        


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
        default="corrupters"
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
        default="jpeg"
    )
    args = parser.parse_args()
    main(args)

    # root_lv1 = os.path.join("Result-Decoded", args.watermarker, args.dataset)
    # corrupter_names = [f for f in os.listdir(root_lv1)]
    # for corrupter in corrupter_names:
    #     root_lv2 = os.path.join(root_lv1, corrupter)
    #     arch_names = [f for f in os.listdir(root_lv2)]
    #     for arch in arch_names:
    #         args.evade_method = corrupter
    #         args.arch = arch
    #         print("Processing: {} - {} - {} - {}".format(args.watermarker, args.dataset, args.evade_method, args.arch))
    #         main(args)
    print("Completed.")