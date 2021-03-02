import os
import json

from . import tf_utils


def finalize(cfg, training=False):
    cfg.SOLVER.NUM_GPUS = len(tf_utils.get_available_gpus())
    cfg.SOLVER.IMS_PER_BATCH = cfg.SOLVER.NUM_GPUS * cfg.SOLVER.IMS_PER_GPU

    with open(os.path.join(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.CATEGORY_MAP_NAME)) as fid:
        category_map = json.load(fid)
    num_thing_classes = category_map["num_thing_classes"]
    num_stuff_classes = category_map["num_stuff_classes"]
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_thing_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_stuff_classes
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = category_map["stuff_ignore_value"]
    cfg.freeze()
    tf_utils.set_training_phase(training=training)
    return cfg