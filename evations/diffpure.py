import argparse, cv2, yaml, torch
import numpy as np
from argparse import ArgumentParser
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from utils.diffpure_utils import GuidedDiffusion, dict2namespace
from utils.general import uint8_to_float, float_to_uint8, img_np_to_tensor, \
    rgb2bgr, bgr2rgb, watermark_np_to_str, compute_bitwise_acc, rgb2bgr, bgr2rgb
import matplotlib.pyplot as plt


class DiffPure():
    """
        Diffpure watermark evasion.
    """
    def __init__(self, steps=0.4, device=torch.device("cuda"), is_stega=False):
        with open('DiffPure/configs/imagenet.yml', 'r') as f:
            config = yaml.safe_load(f)
        self.config = dict2namespace(config)
        self.runner = GuidedDiffusion(self.config, t = int(steps * int(self.config.model.timestep_respacing)), model_dir = 'DiffPure/pretrained/guided_diffusion')
        self.steps = steps
        self.device = device
        self.runner.eval()
        self.is_stega = is_stega

    def forward(self, img):
        img_pured, img_noisy = self.runner.image_editing_sample((img.unsqueeze(0) - 0.5) * 2)
        img_noisy = (img_noisy.squeeze(0).to(img.dtype).to("cpu") + 1) / 2
        img_pured = (img_pured.squeeze(0).to(img.dtype).to("cpu") + 1) / 2
        return img_pured
    
    def __repr__(self):
        return self.__class__.__name__ + '(steps={})'.format(self.steps)
    
    def regenerate(self, im_w_bgr_uint8):
        """
            im_w_bgr_uint8 --- (512, 512, 3)

            NOTE: Although the diffpure model is trained on img resolution 256,
                  the denoise diffuser itself works on 512 resolution too, there's no need for reshaping.
        """
        if not self.is_stega:
            im_w_bgr_reshaped = im_w_bgr_uint8
        else:
            # unfortunately this diffpure cannot work with 400 resolution
            im_w_bgr_reshaped = cv2.resize(im_w_bgr_uint8, (512, 512), interpolation=cv2.INTER_LINEAR)

        img = uint8_to_float(bgr2rgb(im_w_bgr_reshaped))  # [0, 1] np.float32
        img_tensor = img_np_to_tensor(img).squeeze(0).to(self.device)     # [0, 1] torch.tensor
        img_pured = self.forward(img_tensor)

        img_pured_np_float = img_pured.numpy()
        img_pured_bgr_uint8 = rgb2bgr(np.transpose(float_to_uint8(img_pured_np_float), [1, 2, 0]))

        if not self.is_stega:
            img_pured_bgr = img_pured_bgr_uint8
        else:
            img_pured_bgr = cv2.resize(img_pured_bgr_uint8, (400, 400), interpolation=cv2.INTER_AREA)
        return img_pured_bgr

    def regenerate_from_path(self, im_w_path):
        im_w_bgr_uint8 = cv2.imread(im_w_path)
        return self.regenerate(im_w_bgr_uint8)



def diffpure_evation_single_img(
    im_orig_path, im_w_path, watermarker, watermark_gt, evader_cfg=None    
):
    assert evader_cfg is not None, "Must input corruption configs."
    steps = evader_cfg["arch"]
    verbose = evader_cfg["verbose"]
    device = torch.device("cuda")

    watermark_gt_str = watermark_np_to_str(watermark_gt)

    # read images
    im_orig_uint8_bgr = cv2.imread(im_orig_path)
    im_w_uint8_bgr = cv2.imread(im_w_path)

    # Init diffuser
    evader = DiffPure(steps, device)

    # Regnerate
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
        print("  PSNR - <recon v.s im_w> %.02f | <recon v.s clean> %.02f " % (psnr_recon_w, psnr_recon_orig))
        print("  Recon Bitwise Acc. - {:.4f} % ".format(bitwise_acc * 100))
        print("Watermarks: ")
        print("GT:    {}".format(watermark_gt_str))
        print("Recon: {}".format(watermark_recon_str))
    
    return_log = {
        "psnr_w": psnr_recon_w,
        "psnr_clean": psnr_recon_orig,
        "bitwise_acc": bitwise_acc,
        "interm_recon": im_recon_bgr,
    }
    return return_log


def diffpure_interm_collection(im_w_uint8_bgr, evader_cfg=None):
    """
        This function is used to collect all interm. results for large-scale dataset experiments.
    """
    assert evader_cfg is not None, "Must input corruption configs."

    # Init diffuser
    device = torch.device("cuda")
    steps = evader_cfg["arch"]
    is_stega = evader_cfg["is_stegastamp"]

    # Init diffuser
    evader = DiffPure(steps, device, is_stega)

    # Regnerate
    im_recon_bgr = evader.regenerate(im_w_uint8_bgr)

    return_log = {
        "index": [0],
        "interm_recon": [im_recon_bgr]
    }
    return return_log