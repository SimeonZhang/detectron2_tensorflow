import os.path
import copy
import cv2
import numpy as np
import tensorflow as tf
import tqdm

from ..data import fields
from ..structures import np_mask_ops
from . import coco_evaluator
from . import pascal_voc_evaluator
from . import sem_seg_evaluator
from . import panoptic_evaluator

# A dictionary of metric names to classes that implement the metric. The classes
# in the dictionary must implement
# utils.object_detection_evaluation.DetectionEvaluator interface.
_EVAL_METRICS_CLASS_DICT = {
    'coco_detection_metrics':
        coco_evaluator.CocoDetectionEvaluator,
    'coco_instance_segmentation_metrics':
        coco_evaluator.CocoMaskEvaluator,
    'pascal_voc_detection_metrics':
        pascal_voc_evaluator.PascalDetectionEvaluator,
    'weighted_pascal_voc_detection_metrics':
        pascal_voc_evaluator.WeightedPascalDetectionEvaluator,
    'pascal_voc_instance_segmentation_metrics':
        pascal_voc_evaluator.PascalInstanceSegmentationEvaluator,
    'weighted_pascal_voc_instance_segmentation_metrics':
        pascal_voc_evaluator.WeightedPascalInstanceSegmentationEvaluator,
    'semantic_segmentation_metrics':
        sem_seg_evaluator.SemSegEvaluator,
    'panoptic_segmentation_metrics':
        panoptic_evaluator.CocoPanopticEvaluator,
}


def _get_continuous_category_index(
    category_map,
    type="thing",
    load_super_thing_classes=False
):
    dataset_category_index = category_map["category_index"]
    if type == "thing":
        id_map = category_map["thing_id_map"]
    elif type == "stuff":
        id_map = category_map["stuff_id_map"]
    else:
        raise ValueError("Expect either 'thing' or 'stuff', get {}".format(type))
    continuous_category_index = {}
    for idx in id_map:
        if idx in dataset_category_index:
            category = dataset_category_index[idx]
            if load_super_thing_classes:
                continuous_id = category["superid"]
                name = category["supercategory"]
                if isinstance(continuous_id, list):
                    continuous_id = continuous_id[0]
                    name = name[0]
            else:
                continuous_id = id_map[idx]
                name = category.get("display_name")
                if name is None:
                    name = category["name"]
            category["id"] = continuous_id
            category["name"] = name
            continuous_category_index[continuous_id] = category
    return continuous_category_index


class EvaluationHook(tf.train.SessionRunHook):

    def __init__(
        self,
        cfg,
        groundtruth_dict,
        result_dict,
        category_map,
        global_step,
        eval_dir,
        load_super_thing_classes=False,
    ):
        self.groundtruth_dict = groundtruth_dict
        self.result_dict = result_dict
        self.global_step = global_step
        self.process_bar = tqdm.tqdm(total=cfg.EVAL.NUM_EVAL, desc='Eval:')

        if not tf.gfile.Exists(eval_dir):
            tf.gfile.MkDir(eval_dir)
        self.eval_dir = eval_dir
             
        self.segmentation_output_format = cfg.MODEL.SEGMENTATION_OUTPUT.FORMAT
        self.segmentation_output_resolution = cfg.MODEL.SEGMENTATION_OUTPUT.FIXED_RESOLUTION

        self.use_mini_masks = cfg.TRANSFORM.RESIZE.USE_MINI_MASKS

        self.class_agnostic = cfg.EVAL.CLASS_AGNOSTIC

        assert isinstance(cfg.EVAL.METRICS, tuple), cfg.EVAL.METRICS
        self.evaluators = []
        for metric_name in cfg.EVAL.METRICS:
            if metric_name in ['coco_detection_metrics', 'coco_instance_segmentation_metrics']:
                self.evaluators.append(
                    _EVAL_METRICS_CLASS_DICT[metric_name](
                        category_index=_get_continuous_category_index(
                            copy.deepcopy(category_map), type="thing", load_super_thing_classes=load_super_thing_classes),
                        include_metrics_per_category=cfg.EVAL.INCLUDE_METRICS_PER_CATEGORY,
                        all_metrics_per_category=cfg.EVAL.ALL_METRICS_PER_CATEGORY,
                        max_examples_to_draw=cfg.EVAL.MAX_EXAMPLE_TO_DRAW,
                        min_visualization_score_thresh=cfg.EVAL.MIN_VISUALIZATION_SCORE_THRESH
                    )
                )
            elif metric_name in [
                    'pascal_voc_detection_metrics',
                    'weighted_pascal_voc_detection_metrics',
                    'pascal_voc_instance_segmentation_metrics',
                    'weighted_pascal_voc_instance_segmentation_metrics']:
                self.evaluators.append(
                    _EVAL_METRICS_CLASS_DICT[metric_name](
                        category_index=_get_continuous_category_index(
                            copy.deepcopy(category_map), type="thing", load_super_thing_classes=load_super_thing_classes),
                        matching_iou_threshold=cfg.EVAL.PASCAL_MATCHING_IOU_THRESH,
                        max_examples_to_draw=cfg.EVAL.MAX_EXAMPLE_TO_DRAW,
                        min_visualization_score_thresh=cfg.EVAL.MIN_VISUALIZATION_SCORE_THRESH
                    )
                )
            elif metric_name == 'semantic_segmentation_metrics':
                self.evaluators.append(
                    _EVAL_METRICS_CLASS_DICT[metric_name](
                        category_index=_get_continuous_category_index(copy.deepcopy(category_map), type="stuff"),
                        ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                        include_metrics_per_category=cfg.EVAL.INCLUDE_METRICS_PER_CATEGORY,
                        max_examples_to_draw=cfg.EVAL.MAX_EXAMPLE_TO_DRAW,
                    )
                )
            elif metric_name == 'panoptic_segmentation_metrics':
                groundtruth_dir = cfg.DATASETS.ROOT_DIR
                prediction_dir = os.path.join(eval_dir, "panoptic_root")
                self.evaluators.append(
                    _EVAL_METRICS_CLASS_DICT[metric_name](
                        category_map=copy.deepcopy(category_map),
                        stuff_ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                        groundtruth_dir=groundtruth_dir,
                        prediction_dir=prediction_dir,
                        include_metrics_per_category=cfg.EVAL.INCLUDE_METRICS_PER_CATEGORY,
                        all_metrics_per_category=cfg.EVAL.ALL_METRICS_PER_CATEGORY,
                        max_examples_to_draw=cfg.EVAL.MAX_EXAMPLE_TO_DRAW,
                    )
                )
            else:
                raise ValueError("{} is not expected!".format(metric_name))

    def begin(self):
        self.summary_writer = tf.summary.FileWriter(self.eval_dir)
        self.process_bar.reset()

    def before_run(self, run_context):
        fetches = {
            "groundtruth_dict": self.groundtruth_dict,
            "result_dict": self.result_dict,
            "global_step": self.global_step
        }
        return tf.train.SessionRunArgs(fetches=fetches)

    def after_run(self, run_context, run_values):
        input_fields = fields.InputFields
        result_fields = fields.ResultFields

        groundtruth_dict = run_values.results["groundtruth_dict"]
        result_dict = run_values.results["result_dict"]
        self.global_step_value = run_values.results["global_step"]

        input_images = groundtruth_dict[input_fields.image]
        assert len(input_images.shape) == 4, input_images.shape
        num_images = input_images.shape[0]
        self.process_bar.update(num_images)

        for idx in range(num_images):

            gt_per_image, res_per_image = {}, {}
            image_id = groundtruth_dict[fields.InputFields.key][idx].decode()

            true_shape = groundtruth_dict[input_fields.true_shape][idx]
            orig_shape = tuple(groundtruth_dict[input_fields.orig_shape][idx])
            is_valid = groundtruth_dict[input_fields.is_valid][idx]

            for key in groundtruth_dict:
                if key == input_fields.image:
                    res = groundtruth_dict[key][idx][:true_shape[0], :true_shape[1]]
                    # use bilinear method to resize image
                    res = cv2.resize(res, orig_shape[::-1], interpolation=cv2.INTER_LINEAR)
                    gt_per_image[key] = np.round(res).astype(np.uint8)
                elif key == input_fields.sem_seg:
                    res = groundtruth_dict[key][idx][:true_shape[0], :true_shape[1]]
                    res = cv2.resize(res, orig_shape[::-1], interpolation=cv2.INTER_NEAREST)
                    gt_per_image[key] = np.round(res).astype(np.uint8)
                elif key == input_fields.gt_masks:
                    in_masks = groundtruth_dict[key][idx][is_valid, :true_shape[0], :true_shape[1]]
                    num_instances = in_masks.shape[0]
                    out_masks = np.zeros((num_instances,) + orig_shape, dtype=np.uint8)
                    for i in range(num_instances):
                        out_masks[i] = cv2.resize(
                            in_masks[i], orig_shape[::-1], interpolation=cv2.INTER_LINEAR
                        )
                    gt_per_image[key] = np.round(out_masks).astype(np.uint8)
                elif key == input_fields.gt_boxes:
                    boxes = groundtruth_dict[key][idx][is_valid]
                    y_scale = orig_shape[0] / true_shape[0]
                    x_scale = orig_shape[1] / true_shape[1]
                    ymin, xmin, ymax, xmax = np.split(boxes, 4, axis=-1)
                    gt_per_image[key] = np.concatenate(
                        [
                            ymin * y_scale,
                            xmin * x_scale,
                            ymax * y_scale,
                            xmax * x_scale
                        ],
                        axis=-1
                    )
                elif key in [
                    input_fields.gt_classes,
                    input_fields.gt_difficult,
                    input_fields.gt_is_crowd
                ]:
                    gt_per_image[key] = groundtruth_dict[key][idx][is_valid]
                    if self.class_agnostic and key == input_fields.gt_classes:
                        gt_per_image[key] = np.zeros_like(gt_per_image[key])
                else:
                    gt_per_image[key] = groundtruth_dict[key][idx]

            for evaluator in self.evaluators:
                evaluator.add_single_ground_truth_image_info(image_id, gt_per_image)

            if result_fields.sem_seg in result_dict:
                res = result_dict[result_fields.sem_seg][idx]
                if self.segmentation_output_format == "conventional":
                    res = res[:true_shape[0], :true_shape[1]]
                res = cv2.resize(res, orig_shape[::-1], interpolation=cv2.INTER_NEAREST)
                res_per_image[result_fields.sem_seg] = np.round(res).astype(np.uint8)
            if "instances" in result_dict:
                res_per_image["instances"] = {}
                instance_result = result_dict["instances"]
                is_valid = instance_result[result_fields.is_valid][idx]
                boxes = instance_result[result_fields.boxes][idx][is_valid]
                y_scale = orig_shape[0] / true_shape[0]
                x_scale = orig_shape[1] / true_shape[1]
                ymin, xmin, ymax, xmax = np.split(boxes, 4, axis=-1)
                boxes = np.concatenate(
                    [ymin * y_scale, xmin * x_scale, ymax * y_scale, xmax * x_scale], axis=-1
                )
                res_per_image["instances"][result_fields.boxes] = boxes
                for key in instance_result:
                    if key == result_fields.masks:
                        in_masks = instance_result[key][idx][is_valid]
                        if self.segmentation_output_format == "raw":
                            out_masks = np_mask_ops.paste_masks_into_image(in_masks, boxes, orig_shape)
                        else:
                            if self.segmentation_output_format == "conventional":
                                in_masks = in_masks[..., :true_shape[0], :true_shape[1]]
                            num_instances = in_masks.shape[0]
                            out_masks = np.zeros((num_instances,) + orig_shape, dtype=np.uint8)
                            for i in range(num_instances):
                                out_masks[i] = cv2.resize(in_masks[i], orig_shape[::-1], interpolation=cv2.INTER_LINEAR)
                        res_per_image["instances"][key] = np.round(out_masks).astype(np.uint8)
                    elif key == result_fields.boxes:
                        continue
                    elif self.class_agnostic and key == result_fields.classes:
                        res_per_image["instances"][key] = np.zeros_like(instance_result[key][idx][is_valid])
                    else:
                        res_per_image["instances"][key] = instance_result[key][idx][is_valid]
            if "panoptic_seg" in result_dict:
                res_per_image["panoptic_seg"] = {}
                panoptic_result = result_dict["panoptic_seg"]
                is_valid = panoptic_result["is_valid"][idx]
                for key in panoptic_result:
                    if key == "panoptic_seg":
                        res = panoptic_result[key][idx]
                        if self.segmentation_output_format == "conventional":
                            res = res[:true_shape[0], :true_shape[1]]
                        res = cv2.resize(res, orig_shape[::-1], interpolation=cv2.INTER_NEAREST)
                        res_per_image["panoptic_seg"][key] = np.round(res).astype(np.uint32)  # int32 to uint32
                    else:
                        res_per_image["panoptic_seg"][key] = panoptic_result[key][idx][is_valid]
            res_per_image[fields.InputFields.image] = gt_per_image[fields.InputFields.image]
            for evaluator in self.evaluators:
                evaluator.add_single_predicted_image_info(image_id, res_per_image)

    def end(self, session):
        summaries = []
        tf.logging.info("Evaluating...")
        for evaluator in self.evaluators:
            _, summary_values = evaluator.evaluate()
            evaluator.clear()
            summaries.extend(list(summary_values))
        self.summary_writer.add_summary(
            tf.Summary(value=summaries), global_step=self.global_step_value)
        self.summary_writer.close()
