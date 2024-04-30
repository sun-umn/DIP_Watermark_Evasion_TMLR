"""
    Regenerate the image by latent diffusion model to evade the watermark.

    Code adapted from: https://github.com/XuandongZhao/WatermarkAttacker
    
    Major change: I discard the batch mode in order to faithfully evaluate the regeneration quality.

    The original paper actually did a really bad job (at evaluating VAE regenerations, i mean, by downplaying their quality parameters).

    So I don't trust them.

"""
# ## Uncomment Below to run the unit test 
# ########################################
# import sys, os
# dir_path = os.path.abspath(".")
# sys.path.append(dir_path)
# dir_path = os.path.abspath("..")
# sys.path.append(dir_path)
# ###########  ############  #############

import torch
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from diffusers import ReSDPipeline

from utils.general import rgb2bgr
from utils.general import uint8_to_float, float_to_uint8, img_np_to_tensor, \
    tensor_output_to_image_np, watermark_np_to_str, compute_bitwise_acc, rgb2bgr, bgr2rgb


class DiffWMAttacker():
    def __init__(self, pipe, noise_step=60, captions={}):
        self.pipe = pipe
        self.BATCH_SIZE = 1
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions
        print(f'Diffuse attack initialized with noise step {self.noise_step} and use prompt {len(self.captions)}')

    def regenerate_from_path(self, im_w_path):
        with torch.no_grad():
            generator = torch.Generator(self.device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)

            prompt = ""
            img = Image.open(im_w_path)
            img = np.asarray(img) / 255
            img = (img - 0.5) * 2
            img = torch.tensor(img, dtype=torch.float16, device=self.device).permute(2, 0, 1).unsqueeze(0)
            latents = self.pipe.vae.encode(img).latent_dist
            latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor
            noise = torch.randn([1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device)
            latents = self.pipe.scheduler.add_noise(latents, noise, timestep).type(torch.half)
            latents_buf.append(latents)
            prompts_buf.append(prompt)

            latents = torch.cat(latents_buf, dim=0)
            images = self.pipe(prompts_buf,
                                head_start_latents=latents,
                                head_start_step=50 - max(self.noise_step // 20, 1),
                                guidance_scale=7.5,
                                generator=generator, )
            image = images[0][0]
            image_np = np.array(image)
            img_bgr = rgb2bgr(image_np)
            return img_bgr
    
    def regenerate(self, im_w_bgr_uint8):
        with torch.no_grad():
            generator = torch.Generator(self.device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)

            prompt = ""
            img = bgr2rgb(im_w_bgr_uint8).astype(np.float32)
            img = img / 255.
            img = (img - 0.5) * 2
            img = torch.tensor(img, dtype=torch.float16, device=self.device).permute(2, 0, 1).unsqueeze(0)
            latents = self.pipe.vae.encode(img).latent_dist
            latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor
            noise = torch.randn([1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device)
            latents = self.pipe.scheduler.add_noise(latents, noise, timestep).type(torch.half)
            latents_buf.append(latents)
            prompts_buf.append(prompt)

            latents = torch.cat(latents_buf, dim=0)
            images = self.pipe(prompts_buf,
                                head_start_latents=latents,
                                head_start_step=50 - max(self.noise_step // 20, 1),
                                guidance_scale=7.5,
                                generator=generator, )
            image = images[0][0]
            image_np = np.array(image)
            img_bgr = rgb2bgr(image_np)
            return img_bgr


def diffuser_evation_single_img(
    im_orig_path, im_w_path, watermarker, watermark_gt, evader_cfg=None    
):
    """
        Use method proposed in paper: https://arxiv.org/pdf/2306.01953.pdf

        to regenerate a image in order to evade the watermark.

    """
    assert evader_cfg is not None, "Must input corruption configs."
    verbose = evader_cfg["verbose"]
    detection_threshold = evader_cfg["detection_threshold"]

    watermark_gt_str = watermark_np_to_str(watermark_gt)
    # read images
    im_orig_uint8_bgr = cv2.imread(im_orig_path)
    im_w_uint8_bgr = cv2.imread(im_w_path)

    # Init diffuser
    device = torch.device("cuda")
    pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    print('Finished loading model')
    evader = DiffWMAttacker(pipe)

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


def diffuser_interm_collection(im_w_uint8_bgr, evader_cfg=None):
    """
        This function is used to collect all interm. results for large-scale dataset experiments.
    """
    assert evader_cfg is not None, "Must input corruption configs."

    # Init diffuser
    device = torch.device("cuda")
    pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    print('Finished loading model')
    evader = DiffWMAttacker(pipe)

    # Regnerate
    im_recon_bgr = evader.regenerate(im_w_uint8_bgr)

    return_log = {
        "index": [0],
        "interm_recon": [im_recon_bgr]
    }
    return return_log


if __name__ == "__main__":
    print("Unit Test")
    import os
    from diffusers import ReSDPipeline
    test_img_path = os.path.join("examples", "ori_imgs", "000000000711.png")

    # === Change the corrupter you want to test here ===
    device = torch.device("cuda")
    pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    print('Finished loading model')
    evader = DiffWMAttacker(pipe)
    # ===   ===    ===    ===    ===    ===    ===   ===

    img_regenerate = evader.regenerate(test_img_path)
    cv2.imwrite("test_corruption.png", img_regenerate)
    print()