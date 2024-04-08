import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import numpy as np

from .general import save_image_bgr



def plot_dip_res(save_root, res_log):
    # Plot Iter-PSNR curves and PSNR curves
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    iter_data = res_log["iter_log"]
    bw_acc_data = res_log["bitwise_acc"]
    psnr_w_data, psnr_clean_data = res_log["psnr_w"], res_log["psnr_clean"]
    ax[0].plot(iter_data, psnr_clean_data, label="PSNR (recon - clean)")
    ax[0].legend()
    ax[1].plot(iter_data, psnr_w_data, label="PSNR (recon - watermarked)")
    ax[1].legend()
    ax[2].plot(iter_data, bw_acc_data, label="Bitwise Acc.")
    ax[2].legend()
    plt.tight_layout()
    save_name = os.path.join(save_root, "psnr_bt_acc.png")
    plt.savefig(save_name)
    plt.close(fig)

    # Vis Intermediate Recon. Images
    recon_images = res_log["interm_recon"]
    if len(recon_images) < 1:
        print("Do not have interm. images saved. Pass saving visualization.")
    else:
        print("Visualizing Interm. Recon. Images ...")
        save_vis_root = os.path.join(save_root, "Vis-Interm-Recon")
        os.makedirs(save_vis_root, exist_ok=True)
        for idx, iter_num in enumerate(iter_data):
            recon_img = recon_images[idx]
            save_path = os.path.join(
                save_vis_root, "iter-{}.png".format(iter_num)
            )
            save_image_bgr(recon_img, save_path)