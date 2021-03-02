from . import build_coco_det
from . import build_coco_pano


def build(cfg):
    if cfg.BUILD_RECORDS.TYPE == "coco_det":
        build_coco_det.build(cfg)
    elif cfg.BUILD_RECORDS.TYPE == "coco_pano":
        build_coco_pano.build(cfg)
    else:
        raise ValueError("`{:s}` not recognized.".format(cfg.BUILD_RECORDS.TYPE))
