from .iwatermarker import InvisibleWatermarker


def get_watermarkers(cfg):
    watermarker_type = cfg["watermarker"]
    watermarker_gt = cfg["watermark_gt"]
    if watermarker_type in ["rivaGan", "dwtDct", "dwtDctSvd"]:
        watermarker = InvisibleWatermarker(watermarker_type, watermarker_gt)
    else:
        raise RuntimeError("Unsupported watermarker.")

    return watermarker