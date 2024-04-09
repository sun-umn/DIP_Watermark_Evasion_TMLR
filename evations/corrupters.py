import cv2
import numpy as np
from PIL import Image, ImageEnhance
from bm3d import bm3d_rgb

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

from utils.general import uint8_to_float, float_to_uint8, rgb2bgr
from skimage.util import random_noise


class GaussianBlurAttacker():
    def __init__(self, kernel_size=5, sigma=1):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def regenerate(self, im_w_path):
        img_bgr = cv2.imread(im_w_path)
        noisy_img_bgr = cv2.GaussianBlur(img_bgr, (self.kernel_size, self.kernel_size), self.sigma)
        return noisy_img_bgr


class GaussianNoiseAttacker():
    def __init__(self, std=1):
        self.std = std

    def regenerate(self, im_w_path):
        img_bgr = cv2.imread(im_w_path)
        image = uint8_to_float(img_bgr)
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

    def regenerate(self, im_w_path):
        img = Image.open(im_w_path).convert('RGB')
        y_est = bm3d_rgb(np.array(img) / 255, self.std)
        img_bgr = rgb2bgr(y_est)
        img_bgr = float_to_uint8(img_bgr)
        return img_bgr


class JPEGAttacker():
    def __init__(self, quality=80):
        self.quality = quality

    def regenerate(self, im_w_path):
        img_bgr = cv2.imread(im_w_path)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, img_encoded = cv2.imencode('.jpg', img_bgr, encode_param)     
        img_decoded = cv2.imdecode(img_encoded, 1)
        return img_decoded


class BrightnessAttacker():
    def __init__(self, brightness=0.2):
        self.brightness = brightness

    def regenerate(self, im_w_path):
        img = Image.open(im_w_path)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(self.brightness)
        img = rgb2bgr(np.array(img))
        return img


class ContrastAttacker():
    def __init__(self, contrast=0.2):
        self.contrast = contrast

    def regenerate(self, im_w_path):
        img = Image.open(im_w_path)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.contrast)
        img = rgb2bgr(np.array(img))
        return img


if __name__ == "__main__":
    import os
    test_img_path = os.path.join("examples", "ori_imgs", "000000000711.png")

    # === Change the corrupter you want to test here ===
    evader = ContrastAttacker(contrast=10)
    # ===   ===    ===    ===    ===    ===    ===   ===

    img_regenerate = evader.regenerate(test_img_path)
    cv2.imwrite("test_corruption.png", img_regenerate)
    print()