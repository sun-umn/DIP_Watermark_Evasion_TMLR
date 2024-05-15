from PIL import Image
import torch, cv2
import random
import math
import numpy as np
from pytorch_msssim import ssim, ms_ssim
from torchvision import transforms


def set_random_seeds(seed):
    """
        This function sets all random seed used in this experiment.
        For reproduce purpose.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def uint8_to_float(img_orig):
    """
        Convert a uint8 image into float with range [0, 1]
    """
    return img_orig.astype(np.float32) / 255.


def float_to_int(img_float):
    """
        Convert a float image with range [0, 1] into uint 8 with range [0 255]
    """
    return (img_float * 255).round().astype(np.int16)


def float_to_uint8(img_float):
    return (img_float * 255).round().astype(np.uint8)


def img_np_to_tensor(img_np):
    """
        Convert numpy image (float with range [0, 1], shape (N, N, 3)) into tensor input with shape (1, 3, N, N)
    """
    img_np = np.transpose(img_np, [2, 0, 1])
    img_np = img_np[np.newaxis, :, :, :]
    img_tensor = torch.from_numpy(img_np)
    return img_tensor


def tensor_output_to_image_np(out_tensor):
    """
        Convert a tensor output with shape (1, 3, N, N) into float image with shape (3, N, N) and range [0, 1]
    """
    return np.transpose(torch.clamp(out_tensor.detach().cpu(), 0, 1).numpy()[0, :, :, :], [1, 2, 0])


def watermark_np_to_str(watermark_np):
    """
        Convert a watermark in np format into a str to display.
    """
    return "".join([str(i) for i in watermark_np.tolist()])


def watermark_str_to_numpy(watermark_str):
    result = [int(i) for i in watermark_str]
    return np.asarray(result)


def compute_bitwise_acc(watermark_gt, watermark_decoded):
    """
        Compute the bitwise acc., both inputs in ndarray.
    """
    return np.mean(watermark_gt == watermark_decoded)


def bgr2rgb(img_bgr):
    """
        img_bgr.shape = (xx, xx, 3)
    """
    img_rgb = np.stack(
        [img_bgr[:, :, 2], img_bgr[:, :, 1], img_bgr[:, :, 0]], axis=2
    )
    return img_rgb


def rgb2bgr(img_rgb):
    """
        img_bgr.shape = (xx, xx, 3)
    """
    return bgr2rgb(img_rgb)


def save_image_bgr(img_np, path):
    cv2.imwrite(path, img_np)


def save_image_rgb(img_np, path):
    img_np = rgb2bgr(img_np)
    save_image_bgr(img_np, path)



# ==== Adapted from: https://github.com/XuandongZhao/WatermarkAttacker/tree/main ===
def compute_psnr_tensor(a, b):
    """
        Compute the psnr of two tensors with value range [0, 1]
    """
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return 100
    return -10 * math.log10(mse)


def compute_msssim_tensor(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_ssim_tensor(a, b):
    return ssim(a, b, data_range=1.).item()


def eval_psnr_ssim_msssim(ori_img_path, new_img_path):
    ori_img = Image.open(ori_img_path).convert('RGB')
    new_img = Image.open(new_img_path).convert('RGB')
    if ori_img.size != new_img.size:
        new_img = new_img.resize(ori_img.size)
    ori_x = transforms.ToTensor()(ori_img).unsqueeze(0)
    new_x = transforms.ToTensor()(new_img).unsqueeze(0)
    return compute_psnr_tensor(ori_x, new_x), compute_ssim_tensor(ori_x, new_x), compute_msssim_tensor(ori_x, new_x)



def bytearray_to_bits(x):
    """Convert bytearray to a list of bits"""
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def compute_ssim(a, b, data_range):
    """
        Compute the ssim score from 2 cv2 image in np.array.
    """
    a = np.transpose(a, [2, 0, 1])
    a = torch.from_numpy(a).to(dtype=torch.float).unsqueeze(0)
    b = np.transpose(b, [2, 0, 1])
    b = torch.from_numpy(b).to(dtype=torch.float).unsqueeze(0)
    return ssim(a, b, data_range=data_range).item()