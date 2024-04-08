import numpy as np
import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder

from .base import Watermarker    
from utils.general import watermark_np_to_str


class InvisibleWatermarker(Watermarker):
    """
        Realization of DctDwtSVD and RivaGAN using code (pip package)
            from: https://github.com/ShieldMnt/invisible-watermark 
    """
    def __init__(self, method_str, watermark_gt) -> None:
        super().__init__()
        assert method_str in ["rivaGan", "dwtDct", "dwtDctSvd"], "Unsupported watermarker input to Invisible Watermarker."
        print("Initiating ***{}*** encoder & decoder ... ".format(method_str))
        
        watermark_str = watermark_np_to_str(watermark_gt)  # Convert ndarray watermark to string
        print("  GT Watermark - {} \n".format(watermark_str))

        self.method_str = method_str
        self.watermark = watermark_str.encode('utf-8')     # Encode the watermark string into standard format
        self.encoder = WatermarkEncoder()
        self.decoder = WatermarkDecoder("bits", 32)
        self.encoder.set_watermark("bits", self.watermark)
        if method_str == "rivaGan":
            self.encoder.loadModel()
            self.decoder.loadModel()
        
    def encode(self, img_input_path, img_output_path):
        """
            Both input and ouput should be:
                1) ndarray with shape (n, n, 3) 
                2) bgr channel 
                3) uint8
        """
        img_bgr_np = cv2.imread(img_input_path)
        im_w_bgr = self.encoder.encode(img_bgr_np, self.method_str)
        cv2.imwrite(img_output_path, im_w_bgr)

    def decode(self, img_bgr_np):
        """
            Input should be:
                1) ndarray with shape (n, n, 3) 
                2) bgr channel 
                3) uint8
            Output is sequence in ndarray
        """
        decoded = self.decoder.decode(img_bgr_np, self.method_str)
        if self.method_str in ["dwtDct", "dwtDctSvd"]:
            decoded = np.where(decoded == True, 1, 0)
        return decoded


    def decode_from_path(self, img_watermared_path):
        """
            Input should be:
                1) ndarray with shape (n, n, 3) 
                2) bgr channel 
                3) uint8
            Output is sequence in ndarray
        """
        img_bgr_np = cv2.imread(img_watermared_path)
        return self.decode(img_bgr_np)
