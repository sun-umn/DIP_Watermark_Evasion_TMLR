
from DiffPure.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import torch, argparse


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class GuidedDiffusion(torch.nn.Module):
    def __init__(self, config, t, device=None, model_dir='pretrained/guided_diffusion'):
        super().__init__()
        # self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.sample_step = 1
        self.t = t

        # load model
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(self.config.model))
        # print(f'model_config: {model_config}')
        model, diffusion = create_model_and_diffusion(**model_config)
        model.load_state_dict(torch.load(f'{model_dir}/256x256_diffusion_uncond.pt', map_location='cpu'))
        model.requires_grad_(False).eval().to(self.device)

        if model_config['use_fp16']:
            model.convert_to_fp16()

        self.model = model
        self.diffusion = diffusion
        self.betas = torch.from_numpy(diffusion.betas).float().to(self.device)


    def image_editing_sample(self, img, bs_id=0, tag=None):
        with torch.no_grad():
            assert isinstance(img, torch.Tensor)
            batch_size = img.shape[0]

            # if tag is None:
            #     tag = 'rnd' + str(random.randint(0, 10000))
            # out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

            assert img.ndim == 4, img.ndim
            img = img.to(self.device)
            x0 = img

            # if bs_id < 2:
            #     os.makedirs(out_dir, exist_ok=True)
            #     tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'original_input.png'))

            xs = []
            xts = []
            for it in range(self.sample_step):
                e = torch.randn_like(x0)
                total_noise_levels = self.t
                a = (1 - self.betas).cumprod(dim=0)
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                xts.append(x.clone())

                # if bs_id < 2:
                #     tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'init_{it}.png'))

                for i in reversed(range(total_noise_levels)):
                    t = torch.tensor([i] * batch_size, device=self.device)

                    x = self.diffusion.p_sample(self.model, x, t,
                                                clip_denoised=True,
                                                denoised_fn=None,
                                                cond_fn=None,
                                                model_kwargs=None)["sample"]

                    # added intermediate step vis
                    # if (i - 99) % 100 == 0 and bs_id < 2:
                    #     tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'noise_t_{i}_{it}.png'))

                x0 = x

                # if bs_id < 2:
                #     torch.save(x0, os.path.join(out_dir, f'samples_{it}.pth'))
                #     tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'samples_{it}.png'))

                xs.append(x0)

            return torch.cat(xs, dim=0), torch.cat(xts, dim=0)