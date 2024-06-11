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

# ==
from utils.general import watermark_np_to_str, uint8_to_float


watermarkers = [
    "dwtDctSvd",
    "rivaGan",
    "SteganoGAN",
    "SSL",
    "StegaStamp"
]


def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)


def calculate_2dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real


def calc_fft_three_channel(image):
    fft_list = []
    for axis in range(3):
        array = image[:, :, axis]
        fourier = calculate_2dft(array)
        fft_list.append(fourier)
    return np.asarray(fft_list)


def calc_ifft_three_channel(fft_list):
    ifft_list = []
    for axis in range(3):
        array = fft_list[axis]
        ifft = calculate_2dift(array)
        ifft_list.append(ifft)
    return np.stack(ifft_list, axis=2)


def plot_fft(fft_dict, max_res_mag, type_name="", watermarker="none", vis_root=None):
    assert vis_root is not None, "Has to specify the vis root."
    # Visualize Residual Freq maginitude
    for idx in range(3):
        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        fourier = fft_dict[idx]
        magnitude = np.absolute(fourier)

        log_magnitude = np.log(magnitude)
        log_magnitude = np.clip(log_magnitude, -float("inf"), 7)
        min_mag = np.amin(log_magnitude)
        max_mag = np.amax(log_magnitude)
        
        print("Max Mag: ", max_mag, "Min Mag: ", min_mag)
        # log_magnitude = np.clip(log_magnitude, 0, 1)
        log_magnitude = (log_magnitude - min_mag) / (max_mag - min_mag)
        # ax.imshow(log_magnitude, cmap='hot', interpolation='bilinear')
        ax.imshow(log_magnitude)
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        save_name = os.path.join(vis_root, "{}-{}-channel-{}.png".format(watermarker, type_name, idx))
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close(figure)


def main(args):
    dataset = args.dataset
    im_name = args.im_name

    im_clean_path = os.path.join("dataset", "Clean", dataset, im_name)
    im_clean_bgr_uint8 = cv2.imread(im_clean_path)
    im_clean_gray_unit8 = cv2.cvtColor(im_clean_bgr_uint8, cv2.COLOR_BGR2GRAY)
    im_clean_bgr_int = im_clean_bgr_uint8.astype(np.int32)
    im_clean_gray_int = im_clean_gray_unit8.astype(np.int32)
    im_clean_bgr_float = uint8_to_float(im_clean_bgr_uint8)

    max_fourier_mag = -float("inf")  # Use this to normalize the orig fft and watermarked fft
    min_fourier_mag = float("inf")
    # === FFt of clean image ===
    fourier_clean = calc_fft_three_channel(im_clean_bgr_float)
    for i in range(3):
        fft_mag =  np.absolute(fourier_clean[i])
        max_fourier_mag = max(max_fourier_mag, np.amax(fft_mag))
        min_fourier_mag = min(min_fourier_mag, np.amin(fft_mag))

    # === To save all watermarked fft spectrum ===
    fourier_watermark_dict = {}
    fourier_res_dict = {}
    
    # watermarker = "rivaGan"
    for watermarker in watermarkers:
        if watermarker == "StegaStamp":
            im_w_name = im_name.replace(".png", "_hidden.png")
        else:
            im_w_name = im_name
        im_w_path = os.path.join("dataset", watermarker, dataset, "encoder_img", im_w_name)
        if os.path.exists(im_w_path):
            im_w_bgr_uint8 = cv2.imread(im_w_path)
            if watermarker == "StegaStamp":
                im_w_bgr_uint8 = cv2.resize(im_w_bgr_uint8, (512, 512), interpolation=cv2.INTER_CUBIC)
            im_w_gray_uint = cv2.cvtColor(im_w_bgr_uint8, cv2.COLOR_BGR2GRAY)
            im_w_bgr_int = im_w_bgr_uint8.astype(np.int32)
            im_w_gray_int = im_w_gray_uint.astype(np.int32)
        else:
            raise RuntimeError("{} fails to produce a watermarked image {}".format(watermarker, im_w_name))
        
        im_residual_int_bgr = im_w_bgr_int - im_clean_bgr_int  # Err Image
        # Convert the images to float 
        im_w_bgr_float = uint8_to_float(im_w_bgr_uint8)
        im_res_bgr_float = uint8_to_float(im_residual_int_bgr)

        # ==== Check 2d fourier spectrum ====
        # Perform fft2 of the watermarked plot 
        fft_w = calc_fft_three_channel(im_w_bgr_float)
        fourier_watermark_dict[watermarker] = fft_w
        # Update max magnitude
        for i in range(3):
            fft_mag =  np.absolute(fft_w[i])
            max_fourier_mag = max(max_fourier_mag, np.amax(fft_mag))
            min_fourier_mag = min(min_fourier_mag, np.amin(fft_mag))

        # Perform fft2 of the residual plot
        fft_res = calc_fft_three_channel(im_res_bgr_float)
        fourier_res_dict[watermarker] = fft_res
        max_res_mag = -float("inf")
        for i in range(3):
            fft_mag = np.absolute(fft_res[i])
            max_res_mag = max(max_res_mag, np.amax(fft_mag))

        # # ==== Create log folder ====
        vis_root_dir = os.path.join(
            ".", "Vis-Fourier", watermarker
        )
        os.makedirs(vis_root_dir, exist_ok=True)
        # plot_fft(fft_res, max_res_mag, type_name="residual", watermarker=watermarker, vis_root=vis_root_dir)
        plot_fft(fft_res, max_fourier_mag, type_name="residual", watermarker=watermarker, vis_root=vis_root_dir)

        # Plot orig and w
        plot_fft(fourier_clean, max_fourier_mag, type_name="clean", watermarker=watermarker, vis_root=vis_root_dir)
        plot_fft(fft_w, max_fourier_mag, type_name="w", watermarker=watermarker, vis_root=vis_root_dir)

        # # Visualize Residual Freq maginitude
        # for idx in range(3):
        #     figure, ax = plt.subplots(nrows=1, ncols=1)
        #     fourier = fft_res[idx]
        #     magnitude = np.absolute(fourier)
        #     log_magnitude = np.log(magnitude)
        #     ax.imshow(log_magnitude/np.log(max_res_mag), cmap='hot', interpolation='bilinear')
        #     ax.yaxis.grid(False)
        #     ax.xaxis.grid(False)
        #     save_name = os.path.join(vis_root_dir, "{}-residual-channel-{}.png".format(watermarker, idx))
        #     plt.tight_layout()
        #     plt.savefig(save_name)
        #     plt.close(figure)


        # === Calculate Energy Band ====
        test_val = fourier_clean[0]
        center = 255.5
        energy_res_dict = {}
        
        total_dim = 512
        for i in range(total_dim):
            for j in range(total_dim):
                complex_value = test_val[i, j]
                energy = np.absolute(complex_value) ** 2

                radius = int(np.floor(np.sqrt((i-center)**2 + (j-center)**2)))
                if radius not in energy_res_dict.keys():
                    energy_res_dict[radius] = energy
                else:
                    energy_res_dict[radius] += energy

        max_radius = np.amax(list(energy_res_dict.keys()))

        clean_keys, clean_values = np.arange(max_radius+1), np.zeros(max_radius+1)
        for key in energy_res_dict.keys():
            clean_values[key] = energy_res_dict[key]
        max_y = np.amax(clean_values)

        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
        ax.plot(clean_keys, clean_values, label="Clean", color="orange", lw=3, alpha=0.6)
        ax.legend()
        ax.set_ylim([1, max_y*5])
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='y', labelsize=20)
        ax.legend(loc='upper right', ncol=1, fancybox=True, shadow=False, fontsize=20, framealpha=0.3)
        save_name = os.path.join(vis_root_dir, "Clean-energy-band.png")
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close(figure)

    
        # === Calculate Energy Band ====
        test_val = fft_w[0]
        # test_val = fft_res[0]
        center = 255.5
        energy_res_dict = {}
        
        total_dim = 512
        for i in range(total_dim):
            for j in range(total_dim):
                complex_value = test_val[i, j]
                energy = np.absolute(complex_value) ** 2

                radius = int(np.floor(np.sqrt((i-center)**2 + (j-center)**2)))
                if radius not in energy_res_dict.keys():
                    energy_res_dict[radius] = energy
                else:
                    energy_res_dict[radius] += energy

        max_radius = np.amax(list(energy_res_dict.keys()))

        keys, values = np.arange(max_radius+1), np.zeros(max_radius+1)
        for key in energy_res_dict.keys():
            values[key] = energy_res_dict[key]

        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))

        ax.plot(keys, values, label=watermarker, lw=3, alpha=0.6)
        ax.plot(clean_keys, clean_values, color="orange", lw=3, alpha=0.6)
        ax.legend()
        ax.set_ylim([1, max_y*5])
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='y', labelsize=20)
        ax.legend(loc='upper right', ncol=1, fancybox=True, shadow=False, fontsize=20, framealpha=0.3)
        save_name = os.path.join(vis_root_dir, "{}-energy-band.png".format(watermarker))
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close(figure)



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