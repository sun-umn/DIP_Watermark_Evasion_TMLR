import os
import cv2
from torch.utils.data import Dataset
import pandas as pd


class WatermarkedImageDataset(Dataset):
    """
        A customized dataset to make loading the watermarked images with convenience.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        csv_file_path = os.path.join(self.root_dir, "water_mark.csv")
        self.annot_data = pd.read_csv(csv_file_path)

    def __len__(self):
        return len(self.annot_data)

    def __getitem__(self, idx):
        data = self.annot_data.iloc[idx]

        image_name = data["ImageName"]
        image_path = os.path.join(self.root_dir, "encoder_img", image_name)
        image_bgr_np = cv2.imread(image_path)
        watermark_gt_str = data["Encoder"]
        watermark_encoded_str = data["Decoder"]
        # encode_success = data["Match"]
        res = {
            "image_name": image_name.split(".")[0],
            "image_bgr_uint8": image_bgr_np,
            "watermark_gt_str": watermark_gt_str,
            "watermark_encoded_str": watermark_encoded_str
        }
        return res