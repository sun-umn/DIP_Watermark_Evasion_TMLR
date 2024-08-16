import numpy as np
import math, os
import pickle


if __name__ == "__main__":

    # Count File numbers 
    watermarker = "Tree-Ring"
    dataset = "Gustavosta"

    evader = "corrupters"
    archs = ["brightness", "contrast", "gaussian_noise", "jpeg", "bm3d"]
    # archs = ["brightness", "contrast", "gaussian_noise", "jpeg"]
    for arch in archs:
        result_dir = os.path.join(
            "Result-Decoded", watermarker, dataset, evader, arch
        )
        file_list = [f for f in os.listdir(result_dir)]
        print("{} - {} - {} - {}".format(watermarker, dataset, evader, arch))
        print("Number of files processed: ", len(file_list))
        print()
    
    evaders = ["diffuser", "diffpure"]
    arch = "dummy"
    for evader in evaders:
        result_dir = os.path.join(
            "Result-Decoded", watermarker, dataset, evader, arch
        )
        file_list = [f for f in os.listdir(result_dir)]
        print("{} - {} - {} - {}".format(watermarker, dataset, evader, arch))
        print("Number of files processed: ", len(file_list))
        print()

    evader = "vae"
    arch = "cheng2020-anchor"
    result_dir = os.path.join(
        "Result-Decoded", watermarker, dataset, evader, arch
    )
    file_list = [f for f in os.listdir(result_dir)]
    print("{} - {} - {} - {}".format(watermarker, dataset, evader, arch))
    print("Number of files processed: ", len(file_list))
    print()
    
    evader = "dip"
    arch = "vanila"
    result_dir = os.path.join(
        "Result-Decoded", watermarker, dataset, evader, arch
    )
    file_list = [f for f in os.listdir(result_dir)]
    print("{} - {} - {} - {}".format(watermarker, dataset, evader, arch))
    print("Number of files processed: ", len(file_list))
    print()
    print("Completed.")