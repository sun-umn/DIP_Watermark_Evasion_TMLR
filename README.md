# DIP_Watermark_Evasion
This repo contains the source code for paper [PAPER NAME](PAPER_LINK).

If you find our work interesting or helpful, please consider a generous citation of our paper:

```
BIB ITEM TO CITE OUR PAPER
```

## Introduction


## Environment Setup

We recommend using conda env with the following installed

```bash
python==3.12
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then, install the dependencies by:
```bash
pip install -r requirement.txt
```

Finally, run the following command to install the modified [diffusers](https://github.com/huggingface/diffusers) to implement the regeneration attack proposed in [Invisible Image Watermarks Are Provably Removable Using Generative AI](https://arxiv.org/abs/2306.01953).

```bash
pip install -e .
```

### Set up the benchmark from DiffPure from the Official Repo

To config the DiffPure method from [Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks](https://arxiv.org/pdf/2310.00076), first clone the [official repo](https://github.com/mehrdadsaberi/watermark_robustness.git) to PATH_THAT_YOU_LIKE:

```bash
cd PATH_THAT_YOU_LIKE

git clone https://github.com/mehrdadsaberi/watermark_robustness.git
```

Then run the corresponding bash file to download the official pretrained model:

```bash
cd watermark_robustness

bash _bash_download_models.sh
```

Then the entire folder of DiffPure into this repo:

```bash
cp -r DiffPure DIP_Watermark_Evasion/
```


### Troubleshoot for compressai installation

If you have trouble installing ```compressai```, you may trace the code and comment out the parts that uses ```compressai```, which will only affects the vae regeneration benchmarks but not other methods.


## Acknowledgement

Most baseline methods are largely adapted from paper [Invisible Image Watermarks Are Provably Removable Using Generative AI](https://arxiv.org/abs/2306.01953) and their public [code](https://github.com/XuandongZhao/WatermarkAttacker/tree/main).

Baseline "WeVadeBQ" is adapted from paper [Evading Watermark based Detection of AI-Generated Content](https://arxiv.org/abs/2305.03807) and their public [code](https://github.com/zhengyuan-jiang/WEvade).

Only ways to pack the DiffPure method from [Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks](https://arxiv.org/pdf/2310.00076) is provided in this project for convenience. Check the [official repo (full version)](https://github.com/mehrdadsaberi/watermark_robustness.git) for other methods they have proposed.

Special thanks to the original authors and their hard work!