# DIP_Watermark_Evasion
This repo contains the source code for paper [PAPER NAME](PAPER_LINK).

If you find our work interesting or helpful, please consider a generous citation of our paper:

```
BIB ITEM TO CITE OUR PAPER
```

## Introduction


## Environment Setup

First, install the dependencies:
```bash
pip install -r requirement.txt
```
Note that ```compressai``` probably doesn't work on Windows. If you happend to own a windows machine, try using WSL2 and with C++ compliler installed. If you love Windows OS so much, you may have to trace the code and comment out the parts that uses ```compressai```, which will only affects the runability of certain baseline methods (vae generators to be specific).

Then, run the following command to install the modified [diffusers](https://github.com/huggingface/diffusers) to implement the regeneration attack proposed in [Invisible Image Watermarks Are Provably Removable Using Generative AI](https://arxiv.org/abs/2306.01953).

```bash
pip install -e .
```

## Acknowledgement

Most baseline methods are largely adapted from paper [Invisible Image Watermarks Are Provably Removable Using Generative AI](https://arxiv.org/abs/2306.01953) and their public [code](https://github.com/XuandongZhao/WatermarkAttacker/tree/main).

Baseline "WeVadeBQ" is adapted from paper [Evading Watermark based Detection of AI-Generated Content](https://arxiv.org/abs/2305.03807) and their public [code](https://github.com/zhengyuan-jiang/WEvade).

Special thanks to the original authors and their hard work!