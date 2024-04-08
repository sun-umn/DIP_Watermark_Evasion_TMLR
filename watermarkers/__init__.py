from .iwatermarker import InvisibleWatermarker
from .stable_signature import StableSignatureWatermarker


def get_watermarkers(cfg):
    watermarker_type = cfg["watermarker"]
    watermarker_gt = cfg["watermark_gt"]
    if watermarker_type.lower() in ["rivagan", "dwtdct", "dwtdctsvd"]:
        watermarker = InvisibleWatermarker(watermarker_type, watermarker_gt)
    elif watermarker_type.lower() in ["stablesignature"]:
        watermarker = StableSignatureWatermarker()
    else:
        raise RuntimeError("Unsupported watermarker.")

    return watermarker