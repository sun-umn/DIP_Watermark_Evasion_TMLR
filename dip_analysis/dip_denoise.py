"""
    A script tries to explore how DIP denoise the image.
"""

import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import math, torch, cv2, argparse
import numpy as np

# === Project Import ===
from skimage.util import random_noise
from watermarkers import get_watermarkers
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from evations import get_evasion_alg
from utils.plottings import plot_dip_res, plot_vae_res, plot_corruption_res, \
    plot_diffuser_res
from utils.general import watermark_np_to_str, uint8_to_float, img_np_to_tensor, \
    float_to_int, set_random_seeds, float_to_uint8, save_image_bgr, tensor_output_to_image_np
from model_dip import get_net_dip
import matplotlib.pyplot as plt


def main(args):
    # === Some Dummy Configs ===
    set_random_seeds(args.random_seed)
    # === Create Vis Dir === 
    vis_root_dir = os.path.join(
        ".", "Vis-Noisy-Image"
    )
    os.makedirs(vis_root_dir, exist_ok=True)

    img_clean_path = os.path.join(
        args.root_path_im_orig, args.im_name  # Path to a clean image
    )
    img_clean_bgr_uint8_np = cv2.imread(img_clean_path)
    img_clean_bgr_float_np = uint8_to_float(img_clean_bgr_uint8_np)
    # === Add Gaussian Noise ===
    noise_sigma = 0.05
    # noise_sigma = 0.2
    img_noisy_bgr_float_np = random_noise(img_clean_bgr_float_np, mode='gaussian', var=noise_sigma**2)
    # img_noisy_bgr_float_np = random_noise(img_clean_bgr_float_np, mode='s&p', amount=noise_sigma)
    # Below add a constant bias
    img_noisy_bgr_float_np = img_noisy_bgr_float_np
    img_noisy_bgr_float_np = np.clip(img_noisy_bgr_float_np, 0, 1)
    img_noisy_bgr_uint8_np = float_to_uint8(img_noisy_bgr_float_np)

    pure_noise_int_np = img_noisy_bgr_uint8_np.astype(np.int16) - img_clean_bgr_uint8_np.astype(np.int16)
    pure_noise_normed_float = pure_noise_int_np.astype(np.float16)
    value_min, value_max = np.amin(pure_noise_normed_float), np.amax(pure_noise_normed_float)
    print("Noise value range: Min [{}] - Max [{}]".format(value_min, value_max))


    # === Compute Energy ===
    img_clean_energy_mean = np.mean((img_clean_bgr_float_np-np.mean(img_clean_bgr_float_np)) ** 2)
    pure_noise_float = (pure_noise_int_np.astype(np.float32)/255.)
    # noise_energy_mean = np.mean((pure_noise_float - np.mean(pure_noise_float))**2)
    noise_energy_mean = np.mean(pure_noise_float ** 2)
    noise_var = np.mean((pure_noise_float - np.mean(pure_noise_float))**2)
    print("Mean Energy Comparison: ")
    print("   Img - [{}] | Noise - [{}]".format(img_clean_energy_mean, noise_energy_mean))
    print("Noise Variance: - [{:.04f}]".format(noise_var))
    # energy_ratio = img_clean_energy_mean / noise_energy_mean
    # print("   Clean v.s Noise ratio: {:04f}".format(energy_ratio))

    # === Visualize ===
    save_name = os.path.join(vis_root_dir, "img_orig.png")
    save_image_bgr(img_clean_bgr_uint8_np, save_name)
    save_name = os.path.join(vis_root_dir, "img_noisy.png")
    save_image_bgr(img_noisy_bgr_uint8_np, save_name)

    # === Init a DIP to denoise the image ====
    device, dtype = torch.device("cuda"), torch.float
    save_interm = True
    show_every, total_iters = 1, 500
    lr = 0.001
    dip_model = get_net_dip().to(device)
    dip_model.train()
    params = dip_model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_func = torch.nn.MSELoss()

    # == Optimize ===
    iter_log = []
    mse_clean_log = []
    psnr_clean_log = []
    mse_noisy_img_log = []
    psnr_noisy_img_log = []
    recon_interm_log = []

    # === === === ===
    img_noisy_bgr_tensor = img_np_to_tensor(img_noisy_bgr_float_np).to(device, dtype=dtype)
    for num_iter in range(total_iters):
        optimizer.zero_grad()
        net_input = img_noisy_bgr_tensor
        net_output = dip_model(net_input)

        # Compute Loss and Update
        total_loss = loss_func(net_output, img_noisy_bgr_tensor)
        total_loss.backward()
        optimizer.step()

        if num_iter % show_every == 0:
            iter_log.append(num_iter)
            img_recon_float_np = tensor_output_to_image_np(net_output)
            img_recon_int_np = float_to_int(img_recon_float_np)
            img_recon_uint8_np = img_recon_int_np.astype(np.uint8)
            if save_interm:
                recon_interm_log.append(img_recon_uint8_np)

            # Compute component-wise mse (using numpy array in the original scale [0, 255])
            residual_recon = img_recon_float_np - img_clean_bgr_float_np
            # mse and pnsr w.r.t clean image
            mse_orig = np.mean(residual_recon ** 2)
            psnr_orig = compute_psnr(
                img_clean_bgr_uint8_np.astype(np.int16),
                img_recon_int_np,
                data_range=255
            )

            # mse & psnr -- noisy image
            mse_noisy_img = np.mean((img_recon_float_np - img_noisy_bgr_float_np)**2)
            psnr_noisy_img = compute_psnr(
                img_noisy_bgr_uint8_np,
                img_recon_uint8_np,
                data_range=255
            )

            # Log
            mse_clean_log.append(mse_orig)
            psnr_clean_log.append(psnr_orig)
            mse_noisy_img_log.append(mse_noisy_img)
            psnr_noisy_img_log.append(psnr_noisy_img)
            # print("===== Iter [%d] - Loss [%.06f] =====" % (num_iter, total_loss.item()))

    res = {
        "iter_log": iter_log,
        "mse_clean": mse_clean_log,
        "psnr_clean": psnr_clean_log,
        "mse_noisy_img": mse_noisy_img_log,
        "psnr_noisy_img": psnr_noisy_img_log
    }

    # Find the iter with smallest clean mse
    idx = np.argmin(res["mse_clean"])
    mse_clean = res["mse_clean"][idx]
    # mse_noise = res["mse_noise"][idx]
    mse_noisy_img = res["mse_noisy_img"][idx]
    # recon_ratio = mse_noise / mse_clean
    print("MSE comparison at best recon. point: ")
    print("  Clean - [{}] | Noisy Img - [{}]".format(mse_clean, mse_noisy_img))
    # print("  Ratio: {:04f}".format(recon_ratio))

    # === Plot Result ===
    # ITER-MSE curve
    iter_data = res["iter_log"]
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.plot(iter_data, res["mse_clean"], label="recon v.s clean", alpha=0.5, lw=2)
    # ax.hlines(mse_noisy_img, xmin=0, xmax=np.amax(iter_data), label="Best Recon MSE", ls="dashed")
    ax.hlines(noise_energy_mean, xmin=0, xmax=np.amax(iter_data), label="Noise Mean Energy (mse)", ls="dashed", color="red")
    ax.hlines(noise_var, xmin=0, xmax=np.amax(iter_data), label="Noise Variance", ls="dashed")
    ax.plot(iter_data, res["mse_noisy_img"], label="recon v.s. input (noisy image)", alpha=0.5, lw=2)
    ax.vlines(idx, ymin=0, ymax=np.amax(res["mse_noisy_img"]), label="Best Recon Iter", ls="dashed", color="green")
    ax.set_title("Iter - MSE")
    ax.legend()
    plt.tight_layout()
    save_name = os.path.join(vis_root_dir, "plot-mse.png")
    plt.savefig(save_name)
    plt.close(fig)

    # Iter-PSNR Plot
    fig, ax = plt.subplots(ncols=1, nrows=1)
    l1 = ax.plot(iter_data, res["psnr_clean"], label="recon v.s. clean")
    l3 = ax.plot(iter_data, res["psnr_noisy_img"], label="recon v.s. noisy image")
    ax.set_title("Iter - PSNR")
    ax.legend()
    plt.tight_layout()
    save_name = os.path.join(vis_root_dir, "plot-psnr.png")
    plt.savefig(save_name)
    plt.close(fig)

    # === Vis Best Recon ==
    best_recon_img_bgr = recon_interm_log[idx]
    save_name = os.path.join(vis_root_dir, "img_best_recon_psnr_{:.04f}.png".format(res["psnr_clean"][idx]))
    save_image_bgr(best_recon_img_bgr, save_name)

    # === Vis Final Recon ==
    final_recon_img_bgr = recon_interm_log[-1]
    save_name = os.path.join(vis_root_dir, "img_final_recon_psnr_{:.04f}.png".format(res["psnr_clean"][-1]))
    save_image_bgr(final_recon_img_bgr, save_name)
    

if __name__ == "__main__": 

    print("\n***** This is demo of single image evasion ***** \n")
    
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--random_seed", dest="random_seed", type=int, help="Manually set random seed for reproduction.",
        default=42
    )
    parser.add_argument(
        '--root_path_im_orig', type=str, help="Root folder to the clean images.",
        default=os.path.join("examples", "ori_imgs")
    )
    parser.add_argument(
        "--im_name", dest="im_name", type=str, help="clean image name.",
        default="711.png"
    )

    args = parser.parse_args()
    main(args)

    print("\n***** Completed. *****\n")
