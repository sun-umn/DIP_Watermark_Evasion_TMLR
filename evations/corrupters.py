import cv2
import numpy as np
from PIL import Image, ImageEnhance
from bm3d import bm3d_rgb
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

"""
    All tranditional corruption should have a method regenerate which takes in a path input,
    and return a regenerated img in bgr format.

    Code adapted from: https://github.com/XuandongZhao/WatermarkAttacker

    Uncomment the block of code below to run the unit test at the bottom of this script to visualize each corruption.
    You can also trust me that I have tested them.
"""
# #########################################
# import sys, os
# dir_path = os.path.abspath(".")
# sys.path.append(dir_path)
# dir_path = os.path.abspath("..")
# sys.path.append(dir_path)
# ############  ############  #############

from utils.general import uint8_to_float, float_to_uint8, rgb2bgr, bgr2rgb
from skimage.util import random_noise
from utils.general import uint8_to_float, float_to_uint8, \
    watermark_np_to_str, compute_bitwise_acc, rgb2bgr


class GaussianBlurAttacker():
    def __init__(self, sigma=1, kernel_size=5):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def regenerate_from_path(self, im_w_path):
        img_bgr = cv2.imread(im_w_path)
        return self.regenerate(img_bgr)
    
    def regenerate(self, im_w_bgr_unit8):
        noisy_img_bgr = cv2.GaussianBlur(im_w_bgr_unit8, (self.kernel_size, self.kernel_size), self.sigma)
        return noisy_img_bgr


class GaussianNoiseAttacker():
    def __init__(self, std=1):
        self.std = std

    def regenerate_from_path(self, im_w_path):
        img_bgr = cv2.imread(im_w_path)
        return self.regenerate(img_bgr)

    def regenerate(self, im_w_bgr_uint8):
        image = uint8_to_float(im_w_bgr_uint8)
        # Add Gaussian noise to the image
        noise_sigma = self.std  # Vary this to change the amount of noise
        noisy_image = random_noise(image, mode='gaussian', var=noise_sigma ** 2)
        # Clip the values to [0, 1] range after adding the noise
        noisy_image = np.clip(noisy_image, 0, 1)
        noisy_image_bgr = float_to_uint8(noisy_image)
        return noisy_image_bgr


class BM3DAttacker():
    def __init__(self, std=0.1):
        self.std = std  # Comment from the original code: use standard deviation as 0.1, 0.05 also works

    def regenerate_from_file(self, im_w_path):
        img = Image.open(im_w_path).convert('RGB')
        y_est = bm3d_rgb(np.array(img) / 255, self.std)
        img_bgr = rgb2bgr(y_est)
        img_bgr = float_to_uint8(img_bgr)
        return img_bgr

    def regenerate(self, im_w_bgr_uint8):
        img_rgb = bgr2rgb(im_w_bgr_uint8).astype(np.float32)
        y_est = bm3d_rgb(img_rgb / 255., self.std)
        img_bgr = rgb2bgr(y_est)
        img_bgr = float_to_uint8(img_bgr)
        return img_bgr


class JPEGAttacker():
    def __init__(self, quality=0.8):
        self.quality = int(quality * 100)
        print("JPEG compression quality: {:d}".format(self.quality))

    def regenerate_from_file(self, im_w_path):
        img_bgr = cv2.imread(im_w_path)
        return self.regenerate(img_bgr)
    
    def regenerate(self, im_w_bgr_uint8):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, img_encoded = cv2.imencode('.jpg', im_w_bgr_uint8, encode_param)     
        img_decoded = cv2.imdecode(img_encoded, 1)
        return img_decoded


class BrightnessAttacker():
    def __init__(self, brightness=0.2):
        self.brightness = brightness

    def regenerate_from_file(self, im_w_path):
        img = Image.open(im_w_path)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(self.brightness)
        img = rgb2bgr(np.array(img))
        return img
    
    def regenerate(self, im_w_bgr_uint8):
        img = bgr2rgb(im_w_bgr_uint8)
        img = Image.fromarray(img)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(self.brightness)
        img = rgb2bgr(np.array(img))
        return img


class ContrastAttacker():
    def __init__(self, contrast=0.2):
        self.contrast = contrast

    def regenerate_from_file(self, im_w_path):
        img = Image.open(im_w_path)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.contrast)
        img = rgb2bgr(np.array(img))
        return img
    
    def regenerate(self, im_w_bgr_uint8):
        img = bgr2rgb(im_w_bgr_uint8)
        img = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.contrast)
        img = rgb2bgr(np.array(img))
        return img


def get_corrupter(cfg, level=None):
    method_name = cfg["arch"]
    if level is None:
        level = cfg["init_level"]
    
    if method_name.lower() == "gaussian_blur":
        return GaussianBlurAttacker(level)
    elif method_name.lower() == "gaussian_noise":
        return GaussianNoiseAttacker(level)
    elif method_name.lower() == "bm3d":
        return BM3DAttacker(level)
    elif method_name.lower() == "jpeg":
        return JPEGAttacker(level)
    elif method_name.lower() == "brightness":
        return BrightnessAttacker(level)
    elif method_name.lower() == "contrast":
        return ContrastAttacker(level)
    else:
        raise RuntimeError("Un-implemented traditional image corruptions.")


def get_levels(cfg):
    """
        Get Grid search intervals. for different heuristic methods.
    """
    method_name = cfg["arch"]

    if method_name.lower() == "gaussian_blur":
        return np.arange(1, 1001, 10) / 100. 
    elif method_name.lower() == "gaussian_noise":
        return np.arange(1, 101, 1) / 100.
    elif method_name.lower() == "bm3d":
        return np.arange(1, 501, 10) / 100. 
    elif method_name.lower() == "jpeg":
        return np.arange(1, 101, 1) / 100. 
    elif method_name.lower() == "brightness":
        # return np.arange(81, 101, 2) / 100. 
        return np.arange(1, 101, 1) / 100.
    elif method_name.lower() == "contrast":
        # return np.arange(81, 101, 2) / 100. 
        return np.arange(1, 101, 1) / 100.  
    else:
        raise RuntimeError("Un-implemented traditional image corruptions.")
    

def corruption_evation_single_img(
    im_orig_path, im_w_path, watermarker, watermark_gt, evader_cfg=None    
):
    """
        Use traditional image corruption/denoising techniques to evade watermarks.

        Implemented binary search to search for the best possible corruption level.

    """
    assert evader_cfg is not None, "Must input corruption configs."
    verbose = evader_cfg["verbose"]
    detection_threshold = evader_cfg["detection_threshold"]

    watermark_gt_str = watermark_np_to_str(watermark_gt)

    # read images
    im_orig_uint8_bgr = cv2.imread(im_orig_path)
    im_w_uint8_bgr = cv2.imread(im_w_path)

    # === Init the corruption evasion ===
    levels = get_levels(evader_cfg)

    level_log = []
    bitwise_acc_log = []
    psnr_w_log = []
    psnr_clean_log = []
    recon_interm_log = []  # saves the iterm recon result
    best_level, best_psnr = -float("inf"), -float("inf")

    for level in levels:
        evader = get_corrupter(evader_cfg, level)
        im_recon_bgr = evader.regenerate_from_path(im_w_path)

        # === Compute Some Stats ===
        watermark_recon = watermarker.decode(im_recon_bgr)
        watermark_recon_str = watermark_np_to_str(watermark_recon)
        bitwise_acc = compute_bitwise_acc(watermark_gt, watermark_recon)
        # Compute PSNR
        psnr_recon_w = compute_psnr(
            im_w_uint8_bgr, im_recon_bgr, data_range=255  # PSNR of recon v.s. watermarked img
        )
        psnr_recon_orig = compute_psnr(
            im_orig_uint8_bgr, im_recon_bgr, data_range=255  # PSNR of recon v.s. orig
        )

        if verbose:
            print("**** Corrupted corruption level index - [{:04f}]".format(level))
            print("  PSNR - <recon v.s im_w> %.02f | <recon v.s clean> %.02f " % (psnr_recon_w, psnr_recon_orig))
            print("  Recon Bitwise Acc. - {:.4f} % ".format(bitwise_acc * 100))
            print("Watermarks: ")
            print("GT:    {}".format(watermark_gt_str))
            print("Recon: {}".format(watermark_recon_str))

        # === Log Info ===
        level_log.append(level)
        bitwise_acc_log.append(bitwise_acc)
        psnr_w_log.append(psnr_recon_w)
        psnr_clean_log.append(psnr_recon_orig)
        recon_interm_log.append(im_recon_bgr)
        # Update the best recon result
        if psnr_recon_w > best_psnr and bitwise_acc < detection_threshold:
            best_level = level
            best_psnr = psnr_recon_w

    
    return_log = {
        "levels": level_log,
        "psnr_w": psnr_w_log,
        "psnr_clean": psnr_clean_log,
        "bitwise_acc": bitwise_acc_log,
        "interm_recon": recon_interm_log,
        "best_corrupt_level": best_level,
        "best_evade_psnr": best_psnr
    }
    return return_log


def corruption_interm_collection(im_w_uint8_bgr, evader_cfg=None):
    """
        This function is used to collect all interm. results for large-scale dataset experiments.
    """
    assert evader_cfg is not None, "Must input corruption configs."

    # === Init the corruption evasion ===
    levels = get_levels(evader_cfg)

    # Prepare log-info
    index_log = []
    interm_log = []

    for level in levels:
        evader = get_corrupter(evader_cfg, level)
        im_recon_bgr = evader.regenerate(im_w_uint8_bgr)

        index_log.append(level)
        interm_log.append(im_recon_bgr)
    
    return_log = {
        "index": index_log,
        "interm_recon": interm_log
    }
    return return_log


if __name__ == "__main__":
    import os
    test_img_path = os.path.join("examples", "ori_imgs", "000000000711.png")

    # === Change the corrupter you want to test here ===
    evader = ContrastAttacker(contrast=10)
    # ===   ===    ===    ===    ===    ===    ===   ===

    img_regenerate = evader.regenerate_from_file(test_img_path)
    cv2.imwrite("test_corruption.png", img_regenerate)
    print()