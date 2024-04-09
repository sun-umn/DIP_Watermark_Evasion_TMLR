import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor

from utils.general import uint8_to_float, float_to_uint8, img_np_to_tensor, \
    tensor_output_to_image_np, watermark_np_to_str, compute_bitwise_acc, rgb2bgr

class VAEWMAttacker():
    def __init__(self, model_name, quality=1, device=torch.device("cuda")):
        
        if model_name == 'bmshj2018-factorized':
            self.model = bmshj2018_factorized(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'bmshj2018-hyperprior':
            self.model = bmshj2018_hyperprior(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'mbt2018-mean':
            self.model = mbt2018_mean(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'mbt2018':
            self.model = mbt2018(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'cheng2020-anchor':
            self.model = cheng2020_anchor(quality=quality, pretrained=True).eval().to(device)
        else:
            raise ValueError('model name not supported')
        self.device = device
        print("  Use Model - [{}]".format(model_name))

    def regenerate(self, im_w_path):

        img = Image.open(im_w_path).convert('RGB')
        img = img.resize((512, 512))
        img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        out = self.model(img)["x_hat"]
        out = torch.clamp(out, 0, 1)

        out_np = np.transpose(
            out.squeeze().detach().cpu().numpy(),
            [1, 2, 0]
        )
        return out_np
    


def vae_evasion_single_img(
    im_orig_path, im_w_path, watermarker, watermark_gt, evader_cfg=None
):  
    assert evader_cfg is not None, "Must input vae configs."
    device = evader_cfg["device"]
    verbose = evader_cfg["verbose"]
    detection_threshold = evader_cfg["detection_threshold"]
    
    watermark_gt_str = watermark_np_to_str(watermark_gt)
    
    # == Init the vae evasion
    # Set severity range
    evader_method = evader_cfg["arch"]
    quality_range = list(range(1, 7, 1)) if evader_method in ['cheng2020-anchor'] else list(range(1, 9, 1))
    # read images
    im_orig_uint8_bgr = cv2.imread(im_orig_path)
    im_w_uint8_bgr = cv2.imread(im_w_path)

    # == regenerate ==
    quality_log = []
    bitwise_acc_log = []
    psnr_w_log = []
    psnr_clean_log = []
    recon_interm_log = []  # saves the iterm recon result
    best_quality, best_psnr = 0, -float("inf")

    for quality in quality_range:
        evader = VAEWMAttacker(evader_method, quality=quality, device=device)
        im_recon_np_float = evader.regenerate(im_w_path)
        im_recon_rgb = float_to_uint8(im_recon_np_float)
        im_recon_bgr = rgb2bgr(im_recon_rgb)

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
            print("**** VAE regeneration quality index - [{:d}]".format(quality))
            print("  PSNR - <recon v.s im_w> %.02f | <recon v.s clean> %.02f " % (psnr_recon_w, psnr_recon_orig))
            print("  Recon Bitwise Acc. - {:.4f} % ".format(bitwise_acc * 100))
            print("Watermarks: ")
            print("GT:    {}".format(watermark_gt_str))
            print("Recon: {}".format(watermark_recon_str))

        # === Log Info ===
        quality_log.append(quality)
        bitwise_acc_log.append(bitwise_acc)
        psnr_w_log.append(psnr_recon_w)
        psnr_clean_log.append(psnr_recon_orig)
        recon_interm_log.append(im_recon_bgr)
        # Update the best recon result
        if psnr_recon_w > best_psnr and bitwise_acc < detection_threshold:
            best_quality = quality
            best_psnr = psnr_recon_w
    
    return_log = {
        "qualities": quality_log,
        "psnr_w": psnr_w_log,
        "psnr_clean": psnr_clean_log,
        "bitwise_acc": bitwise_acc_log,
        "interm_recon": recon_interm_log,
        "best_evade_quality": best_quality,
        "best_evade_psnr": best_psnr
    }
    return return_log