import os
import pickle
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    file_dir = os.path.join(
        "Img-77.pkl"
    )
    with open(file_dir, 'rb') as handle:
        data_dict = pickle.load(handle)

    interm_recon = data_dict["interm_recon"][0]
    plt.imshow(interm_recon)
    plt.show()
    print()