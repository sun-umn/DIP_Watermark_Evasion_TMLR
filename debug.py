import os
import pickle


if __name__ == "__main__":
    file_dir = os.path.join(
        "Result-Decoded", "SSL", "COCO",
        "vae", "cheng2020-anchor", "Img-1.pkl"
    )
    with open(file_dir, 'rb') as handle:
        data_dict = pickle.load(handle)

    print(data_dict.keys())
    for key in data_dict.keys():
        print("{} - {}".format(key, data_dict[key]))