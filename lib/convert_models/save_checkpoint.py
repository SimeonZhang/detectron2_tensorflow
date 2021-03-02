import pickle
import os
import tensorflow as tf

from ..data import fields
from ..exporter.placeholder import build_batched_input_placeholder
from ..modeling.meta_arch import build_model
from . import convert_d2
from . import convert_backbone
from . import convert_solo
from . import convert_yolov4
from ..utils import tf_utils

tf_utils.set_training_phase(False)


def get_weight_map(cfg):
    if cfg.PRETRAINS.DARKNET:
        return convert_yolov4.get_weight_map(cfg)
    if cfg.PRETRAINS.DETECTRON2:
        weight_path = os.path.join(cfg.PRETRAINS.ROOT, cfg.PRETRAINS.DETECTRON2)
        convert_fn = convert_d2.convert_weights
    elif cfg.PRETRAINS.BACKBONE:
        weight_path = os.path.join(cfg.PRETRAINS.ROOT, cfg.PRETRAINS.BACKBONE)
        convert_fn = convert_backbone.convert_weights
    elif cfg.PRETRAINS.MMDET:
        weight_path = os.path.join(cfg.PRETRAINS.ROOT, cfg.PRETRAINS.MMDET)
        convert_fn = convert_solo.convert_weights
    else:
        raise ValueError("d2 or msra model path must be given.")

    with open(weight_path, "rb") as fid:
        weight_map = pickle.load(fid)
        if "model" in weight_map:
            weight_map = weight_map["model"]
        if "blobs" in weight_map:
            weight_map = weight_map["blobs"]
    weight_map = convert_fn(weight_map, cfg)
    return weight_map


def save(cfg):
    weight_map = get_weight_map(cfg)
    build_model(cfg)
    scope="backbone" if cfg.PRETRAINS.BACKBONE else None 
    model_variables = tf.model_variables(scope=scope)

    assign_ops = []
    for var in model_variables:
        if var.op.name in weight_map:
            assign_ops.append(tf.assign(var, weight_map.pop(var.op.name)))
        else:
            print(var.op.name, " not found.")

    for var_name in weight_map:
        print(var_name, " is not assigned.")

    saver = tf.train.Saver(model_variables)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(assign_ops)
    # Save the variable to the disk
    checkpoint_path = os.path.join(cfg.PRETRAINS.ROOT, cfg.PRETRAINS.WEIGHTS)
    saver.save(sess, checkpoint_path, write_meta_graph=False, write_state=False)
