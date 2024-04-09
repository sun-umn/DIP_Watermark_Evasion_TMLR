"""
    Regenerate the image by latent diffusion model to evade the watermark.

    Code adapted from: https://github.com/XuandongZhao/WatermarkAttacker
    
    Major change: I discard the batch mode in order to faithfully evaluate the regeneration quality.

    The original paper actually did a really bad job (at evaluating VAE regenerations, i mean, by downplaying their quality parameters).

    So I don't trust them.

"""
import torch
import numpy as np
from PIL import Image

class DiffWMAttacker():
    def __init__(self, pipe, noise_step=60, captions={}):
        self.pipe = pipe
        self.BATCH_SIZE = 1
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions
        print(f'Diffuse attack initialized with noise step {self.noise_step} and use prompt {len(self.captions)}')

    def regenerate(self, im_w_path, return_latents=False, return_dist=False):
        with torch.no_grad():
            generator = torch.Generator(self.device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            outs_buf = []
            timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)
            ret_latents = []

            # def batched_attack(latents_buf, prompts_buf, outs_buf):
            #     latents = torch.cat(latents_buf, dim=0)
            #     images = self.pipe(prompts_buf,
            #                        head_start_latents=latents,
            #                        head_start_step=50 - max(self.noise_step // 20, 1),
            #                        guidance_scale=7.5,
            #                        generator=generator, )
            #     images = images[0]
            #     for img, out in zip(images, outs_buf):
            #         img.save(out)

            prompt = ""
            img = Image.open(im_w_path)
            img = np.asarray(img) / 255
            img = (img - 0.5) * 2
            img = torch.tensor(img, dtype=torch.float16, device=self.device).permute(2, 0, 1).unsqueeze(0)
            latents = self.pipe.vae.encode(img).latent_dist
            latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor
            noise = torch.randn([1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device)
            if return_dist:
                return self.pipe.scheduler.add_noise(latents, noise, timestep, return_dist=True)
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
            return image
        

if __name__ == "__main__":
    print("Unit Test")