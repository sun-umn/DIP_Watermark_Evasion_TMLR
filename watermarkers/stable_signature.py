import subprocess, os, torch
from torchvision import transforms
from utils.general import watermark_np_to_str
from PIL import Image

from .base import Watermarker



class StableSignatureWatermarker(Watermarker):
    def __init__(
            self, stable_diffusion_root_path, msg_extractor, 
            script, watermark_gt, device=torch.device("cuda")
        ):
        print("Initiating *** Stable Signiture *** encoder & decoder ... ")
        self.stable_diffusion_root_path = stable_diffusion_root_path
        self.key = watermark_np_to_str(watermark_gt)
        print("  GT Watermark - {} \n".format(self.key))

        self.device = device
        self.msg_extractor = torch.jit.load(msg_extractor).to(self.device)
        self.transform_imnet = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.script = script

    def encode(self, img_path, output_path, prompt=''):
        command = [
            'python', os.path.join(self.stable_diffusion_root_path, f'scripts/{self.script}'),
            '--prompt', prompt,
            '--ckpt', os.path.join(self.stable_diffusion_root_path, 'checkpoints/v2-1_512-ema-pruned.ckpt'),
            '--config', os.path.join(self.stable_diffusion_root_path, 'configs/stable-diffusion/v2-inference.yaml'),
            '--H', '512',
            '--W', '512',
            '--device', 'cuda',
            '--outdir', output_path,
            '--img_name', img_path,
            '--n_samples', '1',
            '--n_rows', '1',
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        # Print the output or handle error
        if result.returncode != 0:
            print('Error:', result.stderr)
        else:
            print('Output:', result.stdout)

    def decode(self, img_path):
        img = Image.open(img_path)
        img = self.transform_imnet(img).unsqueeze(0).to(self.device)
        msg = self.msg_extractor(img)  # b c h w -> b k
        # bool_msg = (msg > 0).squeeze().cpu().numpy().tolist()
        bool_msg = (msg > 0).squeeze().cpu().numpy()
        return bool_msg
    
        # bool_key = StableSignatureWatermarker.str2msg(self.key)
        # # compute difference between model key and message extracted from image
        # diff = [bool_msg[i] != bool_key[i] for i in range(len(bool_msg))]
        # bit_acc = 1 - sum(diff) / len(diff)
        # print("Bit accuracy: ", bit_acc)

        # # compute p-value
        # from scipy.stats import binomtest
        # pval = binomtest(sum(diff), len(diff), 0.5)
        # print("p-value of statistical test: ", pval)
        # return bit_acc, pval

    # def msg2str(msg):
    #     return "".join([('1' if el else '0') for el in msg])

    # def str2msg(str):
    #     return [True if el == '1' else False for el in str]