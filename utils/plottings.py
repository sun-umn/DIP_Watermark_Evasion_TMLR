import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import numpy as np

from .general import save_image_bgr


def plot_dip_res(save_root, res_log, detection_threshold=0.75, vis_recon=False):
    # Plot Iter-PSNR curves and bitwise acc.
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    iter_data = res_log["iter_log"]
    bw_acc_data = res_log["bitwise_acc"]
    psnr_w_data, psnr_clean_data = res_log["psnr_w"], res_log["psnr_clean"]
    ax[0].plot(iter_data, psnr_clean_data, label="PSNR (recon - clean)", color="orange")
    ax[0].plot(iter_data, psnr_w_data, label="PSNR (recon - watermarked)", color="blue", ls="dashed")
    ax[0].vlines(res_log["best_evade_iter"], ymin=np.amin(psnr_w_data), ymax=np.amax(psnr_w_data), color="black", ls="dashed", label="Best Recon Iter")
    ax[0].legend()
    ax[1].plot(iter_data, bw_acc_data, label="Bitwise Acc.")
    ax[1].hlines(y=detection_threshold, xmin=np.amin(iter_data), xmax=np.amax(iter_data), ls="dashed", color="black")
    ax[1].hlines(y=(1-detection_threshold), xmin=np.amin(iter_data), xmax=np.amax(iter_data), ls="dashed", color="black")
    ax[1].vlines(res_log["best_evade_iter"], ymin=0, ymax=1, color="black", ls="dashed", label="Best Recon Iter")
    ax[1].legend()
    plt.tight_layout()
    save_name = os.path.join(save_root, "psnr_bt_acc.png")
    plt.savefig(save_name)
    plt.close(fig)

    # Vis Intermediate Recon. Images
    recon_images = res_log["interm_recon"]
    if len(recon_images) < 1:
        print("Do not have interm. images saved. Pass saving visualization.")
    elif not vis_recon:
        print("Opt out to visualize the intermediate reconstructed images.")
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

    # Vis Component-wise mse
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    iter_data = res_log["iter_log"]
    ax[0].plot(iter_data, res_log["mse_to_orig"], label="MSE (recon - clean)")
    ax[0].plot(iter_data, res_log["mse_to_watermark"], label="MSE (recon - im_w)")
    ax[0].hlines(res_log["best_evade_mse"], xmin=0, xmax=np.amax(iter_data), label="Best evade MSE (recon v.s. im_w)", ls="dashed", color="orange")
    ax[0].hlines(res_log["mse_clean_to_w"], xmin=0, xmax=np.amax(iter_data), label="MSE (clean v.s. im_w)", ls="dashed", color="black")
    ax[0].vlines(res_log["best_evade_iter"], ymin=0, ymax=np.amax(res_log["mse_to_watermark"]), color="black", ls="dashed", label="Best Recon Iter")
    ax[0].legend()
    ax[0].set_yscale('log')
    ax[1].plot(iter_data, bw_acc_data, label="Bitwise Acc.")
    ax[1].hlines(y=detection_threshold, xmin=np.amin(iter_data), xmax=np.amax(iter_data), ls="dashed", color="black")
    ax[1].hlines(y=(1-detection_threshold), xmin=np.amin(iter_data), xmax=np.amax(iter_data), ls="dashed", color="black")
    ax[1].vlines(res_log["best_evade_iter"], ymin=0, ymax=1, color="black", ls="dashed", label="Best Recon Iter")
    ax[1].legend()
    save_name = os.path.join(save_root, "MSE_plot.png")
    plt.savefig(save_name)
    plt.close(fig)
    

def plot_vae_res(save_root, res_log, detection_threshold=0.75):
    # Plot Quality-PSNR curves and bitwise acc. curves
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    quality_data = res_log["qualities"]
    bw_acc_data = res_log["bitwise_acc"]
    psnr_w_data, psnr_clean_data = res_log["psnr_w"], res_log["psnr_clean"]
    ax[0].plot(quality_data, psnr_clean_data, label="PSNR (recon - clean)")
    ax[0].legend()
    ax[1].plot(quality_data, psnr_w_data, label="PSNR (recon - watermarked)")
    ax[1].legend()
    ax[2].plot(quality_data, bw_acc_data, label="Bitwise Acc.")
    ax[2].hlines(y=detection_threshold, xmin=np.amin(quality_data), xmax=np.amax(quality_data), ls="dashed", color="black")
    ax[2].hlines(y=(1-detection_threshold), xmin=np.amin(quality_data), xmax=np.amax(quality_data), ls="dashed", color="black")
    ax[2].legend()
    ax[2].set_xlabel("VAE regeneration quality index")
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
        save_vis_root = os.path.join(save_root, "Vis-Recon-PerQuality")
        os.makedirs(save_vis_root, exist_ok=True)
        for idx, quality_number in enumerate(quality_data):
            recon_img = recon_images[idx]
            save_path = os.path.join(
                save_vis_root, "Quality-{}.png".format(quality_number)
            )
            save_image_bgr(recon_img, save_path)



def plot_corruption_res(save_root, res_log, detection_threshold=0.75, method_name=None):
    if "jpeg" in method_name.lower():
        factor = 100  # level factor modifier
    else:
        factor = 1

    # Plot Corr_level-PSNR curves and bitwise acc. curves
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    level_data = np.asarray(res_log["levels"]) * factor
    bw_acc_data = res_log["bitwise_acc"]
    psnr_w_data, psnr_clean_data = res_log["psnr_w"], res_log["psnr_clean"]
    ax[0].scatter(level_data, psnr_clean_data, label="PSNR (recon - clean)")
    ax[0].legend()
    ax[1].scatter(level_data, psnr_w_data, label="PSNR (recon - watermarked)")
    ax[1].legend()
    ax[2].scatter(level_data, bw_acc_data, label="Bitwise Acc.")
    ax[2].hlines(y=detection_threshold, xmin=np.amin(level_data), xmax=np.amax(level_data), ls="dashed", color="black")
    ax[2].hlines(y=(1-detection_threshold), xmin=np.amin(level_data), xmax=np.amax(level_data), ls="dashed", color="black")
    ax[2].legend()
    ax[2].set_xlabel("{} level".format(method_name))
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
        save_vis_root = os.path.join(save_root, "Vis-Recon-PerQuality")
        os.makedirs(save_vis_root, exist_ok=True)
        for idx, level_number in enumerate(level_data):
            recon_img = recon_images[idx]
            save_path = os.path.join(
                save_vis_root, "Level-{}.png".format(level_number)
            )
            save_image_bgr(recon_img, save_path)


def plot_diffuser_res(save_root, res_log):
    save_vis_root = os.path.join(save_root, "Vis-Recon-PerQuality")
    os.makedirs(save_vis_root, exist_ok=True)
    save_path = os.path.join(
        save_vis_root, "Diffuser_regenerated.png"
    )
    recon_img = res_log["interm_recon"]
    save_image_bgr(recon_img, save_path)