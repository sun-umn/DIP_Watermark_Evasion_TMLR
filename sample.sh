
python msi_experiments/collect_evade_interm_results.py --watermarker SSL --dataset COCO --evade_method vae --arch cheng2020-anchor --random_seed 13 --start 0  &> logs/SSL_COCO.txt
python msi_experiments/collect_evade_interm_results.py --watermarker SSL --dataset DiffusionDB --evade_method vae --arch cheng2020-anchor --random_seed 13 --start 0 &> logs/SSL_DiffusionDB.txt