class Watermarker:
    def encode(self, img_bgr_np):
        """
            The input are standardized by:
            1) ndarray with shape (n, n, 3)
            2) bgr channel
            3) uint8
        """
        raise NotImplementedError

    def decode(self, img_bgr_np):
        """
            The input are standardized by:
            1) ndarray with shape (n, n, 3)
            2) bgr channel
            3) uint8
        """
        raise NotImplementedError