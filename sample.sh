
# python playgrounds/playground_single_image.py --evade_method corrupters --arch gaussian_noise
# python playgrounds/playground_single_image.py --evade_method corrupters --arch gaussian_blur
# python playgrounds/playground_single_image.py --evade_method corrupters --arch jpeg
# python playgrounds/playground_single_image.py --evade_method corrupters --arch brightness
# python playgrounds/playground_single_image.py --evade_method corrupters --arch contrast
# python playgrounds/playground_single_image.py --evade_method corrupters --arch bm3d


# python msi_experiments/collect_evade_interm_results.py --watermarker dwtDctSvd --dataset COCO --evade_method vae --arch cheng2020-anchor
# python msi_experiments/collect_evade_interm_results.py --watermarker dwtDctSvd --dataset DiffusionDB --evade_method vae --arch cheng2020-anchor
# python msi_experiments/collect_evade_interm_results.py --watermarker rivaGan --dataset COCO --evade_method vae --arch cheng2020-anchor
# python msi_experiments/collect_evade_interm_results.py --watermarker rivaGan --dataset DiffusionDB --evade_method vae --arch cheng2020-anchor
# python msi_experiments/collect_evade_interm_results.py --watermarker SSL --dataset COCO --evade_method vae --arch cheng2020-anchor
# python msi_experiments/collect_evade_interm_results.py --watermarker SSL --dataset DiffusionDB --evade_method vae --arch cheng2020-anchor
# python msi_experiments/collect_evade_interm_results.py --watermarker SteganoGAN --dataset COCO --evade_method vae --arch cheng2020-anchor
# python msi_experiments/collect_evade_interm_results.py --watermarker SteganoGAN --dataset DiffusionDB --evade_method vae --arch cheng2020-anchor
python msi_experiments/collect_evade_interm_results.py --watermarker StegaStamp --dataset COCO --evade_method vae --arch cheng2020-anchor
python msi_experiments/collect_evade_interm_results.py --watermarker StegaStamp --dataset DiffusionDB --evade_method vae --arch cheng2020-anchor

# python msi_experiments/collect_evade_interm_results.py --watermarker dwtDctSvd --dataset COCO --evade_method vae --arch mbt2018
# python msi_experiments/collect_evade_interm_results.py --watermarker dwtDctSvd --dataset DiffusionDB --evade_method vae --arch mbt2018
# python msi_experiments/collect_evade_interm_results.py --watermarker rivaGan --dataset COCO --evade_method vae --arch mbt2018
# python msi_experiments/collect_evade_interm_results.py --watermarker rivaGan --dataset DiffusionDB --evade_method vae --arch mbt2018
# python msi_experiments/collect_evade_interm_results.py --watermarker SSL --dataset COCO --evade_method vae --arch mbt2018
# python msi_experiments/collect_evade_interm_results.py --watermarker SSL --dataset DiffusionDB --evade_method vae --arch mbt2018
# python msi_experiments/collect_evade_interm_results.py --watermarker SteganoGAN --dataset COCO --evade_method vae --arch mbt2018
# python msi_experiments/collect_evade_interm_results.py --watermarker SteganoGAN --dataset DiffusionDB --evade_method vae --arch mbt2018
python msi_experiments/collect_evade_interm_results.py --watermarker StegaStamp --dataset COCO --evade_method vae --arch mbt2018
python msi_experiments/collect_evade_interm_results.py --watermarker StegaStamp --dataset DiffusionDB --evade_method vae --arch mbt2018

# python msi_experiments/collect_evade_interm_results.py --watermarker dwtDctSvd --dataset COCO --evade_method vae --arch bmshj2018-factorized
# python msi_experiments/collect_evade_interm_results.py --watermarker dwtDctSvd --dataset DiffusionDB --evade_method vae --arch bmshj2018-factorized
# python msi_experiments/collect_evade_interm_results.py --watermarker rivaGan --dataset COCO --evade_method vae --arch bmshj2018-factorized
# python msi_experiments/collect_evade_interm_results.py --watermarker rivaGan --dataset DiffusionDB --evade_method vae --arch bmshj2018-factorized
# python msi_experiments/collect_evade_interm_results.py --watermarker SSL --dataset COCO --evade_method vae --arch bmshj2018-factorized
# python msi_experiments/collect_evade_interm_results.py --watermarker SSL --dataset DiffusionDB --evade_method vae --arch bmshj2018-factorized
# python msi_experiments/collect_evade_interm_results.py --watermarker SteganoGAN --dataset COCO --evade_method vae --arch bmshj2018-factorized
# python msi_experiments/collect_evade_interm_results.py --watermarker SteganoGAN --dataset DiffusionDB --evade_method vae --arch bmshj2018-factorized
# python msi_experiments/collect_evade_interm_results.py --watermarker StegaStamp --dataset COCO --evade_method vae --arch bmshj2018-factorized
# python msi_experiments/collect_evade_interm_results.py --watermarker StegaStamp --dataset DiffusionDB --evade_method vae --arch bmshj2018-factorized