import os
import tensorflow as tf
import json

from ..data import fields
from ..data.dataloader import build_dataloader
from ..modeling.meta_arch import build_model
from ..structures import box_list
from ..structures import box_list_ops
from ..structures import mask_ops
from ..evaluation import evaluation

slim = tf.contrib.slim


def extract_groundtruths_and_predictions(cfg):
    input_fields = fields.InputFields
    dataloader = build_dataloader(cfg, training=False)
    tensor_dict = dataloader.get_next()

    images = tensor_dict[input_fields.image]
    batched_inputs = {
        'image': images,
        'image_shape': tensor_dict[input_fields.true_shape],
    }

    if input_fields.sem_seg in tensor_dict:
        batched_inputs['sem_seg'] = tensor_dict[input_fields.sem_seg]

    instances = box_list.BoxList(tensor_dict[input_fields.gt_boxes])
    instances.add_field('gt_classes', tensor_dict[input_fields.gt_classes])
    instances.add_field('gt_difficult', tensor_dict[input_fields.gt_difficult])
    instances.add_field('gt_is_crowd', tensor_dict[input_fields.gt_is_crowd])
    instances.add_field('is_valid', tensor_dict[input_fields.is_valid])
    if input_fields.gt_masks in tensor_dict:
        instances.add_field('gt_masks', tensor_dict[input_fields.gt_masks])
    batched_inputs['instances'] = instances

    model_fn = build_model(cfg)
    results = model_fn(batched_inputs)

    if cfg.TRANSFORM.RESIZE.USE_MINI_MASKS:
        if input_fields.gt_masks in tensor_dict:
            image_shape = tf.shape(images)[1:3]
            sparse_instances = box_list.SparseBoxList.from_dense(instances)
            boxes = sparse_instances.data.boxes
            gt_masks = sparse_instances.data.get_field("gt_masks")
            gt_masks = mask_ops.reframe_box_masks_to_image_masks(gt_masks, boxes, image_shape)
            sparse_instances.data.set_field("gt_masks", gt_masks)
            instances = sparse_instances.to_dense()
            tensor_dict[input_fields.gt_masks] = instances.get_field("gt_masks")

    return tensor_dict, results


def evaluate(cfg):
    eval_dir = os.path.join(cfg.LOGS.ROOT_DIR, cfg.LOGS.EVAL)
    checkpoint_dir = os.path.join(cfg.LOGS.ROOT_DIR, cfg.LOGS.TRAIN)
    with open(os.path.join(cfg.DATASETS.ROOT_DIR, "category_map.json")) as fid:
        category_map = json.load(fid)

    graph = tf.Graph()
    with graph.as_default():

        groundtruth_dict, result_dict = extract_groundtruths_and_predictions(cfg)

        eval_hook = evaluation.EvaluationHook(
            cfg,
            groundtruth_dict=groundtruth_dict,
            result_dict=result_dict,
            category_map=category_map,
            global_step=slim.get_or_create_global_step(),
            eval_dir=eval_dir
        )
        eval_op = {}
        eval_op.update(result_dict)
        writer = tf.summary.FileWriter(logdir=eval_dir, graph=graph)
        writer.close()
        num_evals = cfg.EVAL.NUM_EVAL // cfg.SOLVER.IMS_PER_GPU
        slim.evaluation.evaluation_loop(
            master="",
            checkpoint_dir=checkpoint_dir,
            num_evals=num_evals,
            eval_op=eval_op,
            logdir=eval_dir,
            hooks=[eval_hook],
            variables_to_restore=slim.get_variables_to_restore())
