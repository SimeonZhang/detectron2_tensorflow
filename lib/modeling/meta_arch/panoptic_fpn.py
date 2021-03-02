from collections import defaultdict
import tensorflow as tf

from ...data import fields
from ...layers import Layer
from ...structures import ImageList
from ..backbone import build_backbone
from ..necks import build_neck
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from ..postprocessing import detector_postprocess, sem_seg_postprocess
from .build import META_ARCH_REGISTRY
from .semantic_seg import build_sem_seg_head


@META_ARCH_REGISTRY.register()
class PanopticFPN(Layer):
    """
    Main class for Panoptic FPN architectures (see https://arxiv.org/abd/1901.02446).
    """

    def __init__(self, cfg):
        super().__init__()

        self.instance_loss_weight = cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT

        # options when combining instance & semantic outputs
        self.combine_on = cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED
        self.combine_overlap_threshold = cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH
        self.combine_stuff_area_limit = cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT
        self.combine_instances_confidence_threshold = (
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH
        )

        self.backbone = build_backbone(cfg, scope="backbone")
        self.neck = build_neck(cfg, self.backbone.output_shape(), scope="neck")
        self.proposal_generator = build_proposal_generator(
            cfg, self.neck.output_shape(), scope="proposal_generator")
        self.roi_heads = build_roi_heads(cfg, self.neck.output_shape(), scope="roi_heads")
        self.sem_seg_head = build_sem_seg_head(
            cfg, self.neck.output_shape(), scope="sem_seg_head"
        )

        pixel_mean = tf.convert_to_tensor(cfg.MODEL.PIXEL_MEAN, tf.float32)
        pixel_std = tf.convert_to_tensor(cfg.MODEL.PIXEL_STD, tf.float32)
        self.input_format = cfg.MODEL.INPUT_FORMAT
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.segmentation_output_format = cfg.MODEL.SEGMENTATION_OUTPUT.FORMAT
        self.segmentation_output_resolution = cfg.MODEL.SEGMENTATION_OUTPUT.FIXED_RESOLUTION
        assert self.segmentation_output_format != "raw", "Must transform box masks to image masks."

    def call(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            instances: Instances
            sem_seg: semantic segmentation ground truth.
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            list[dict]: each dict is the results for one image. The dict
                contains the following keys:
                "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
                    See the return value of
                    :func:`combine_semantic_and_instance_outputs` for its format.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = self.neck(features)

        if "proposals" in batched_inputs:
            proposals = batched_inputs["proposals"]
            proposal_losses = {}

        if "sem_seg" in batched_inputs:
            gt_sem_seg = batched_inputs["sem_seg"]
            gt_sem_seg = ImageList.from_tensors(
                gt_sem_seg,
                image_shapes,
                self.backbone.size_divisibility,
                self.sem_seg_head.ignore_value
            ).tensor
        else:
            gt_sem_seg = None
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)

        if "instances" in batched_inputs:
            gt_instances = batched_inputs["instances"]
        else:
            gt_instances = None
        if self.proposal_generator:
            proposals, proposal_losses, _ = self.proposal_generator(images, features, gt_instances)
        detected_instances, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        if self.training:
            losses = {}
            losses.update(sem_seg_losses)
            losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
            losses.update(proposal_losses)
            return losses

        if self.segmentation_output_format == "fixed":
            output_shape = [self.output_resolution, self.output_resolution]
        elif self.segmentation_output_format == "conventional":
            output_shape = tf.shape(images.tensor)[1:3]
        detected_instances = detector_postprocess(
            detected_instances, output_shape, self.segmentation_output_format, images.image_shapes
        )
        sem_seg_results = sem_seg_postprocess(
            sem_seg_results, images.image_shapes, output_shape, self.segmentation_output_format
        )

        results = {}
        result_fields = fields.ResultFields
        results["instances"] = {
            result_fields.boxes: detected_instances.boxes,
            result_fields.classes: detected_instances.get_field("pred_classes"),
            result_fields.scores: detected_instances.get_field("scores"),
            result_fields.is_valid: detected_instances.get_field("is_valid"),
            result_fields.masks: detected_instances.get_field("pred_masks")
        }
        sem_seg_results = tf.argmax(sem_seg_results, axis=-1)
        results[result_fields.sem_seg] = sem_seg_results
        if self.combine_on:
            segments_info = combine_semantic_and_instance_outputs(
                results["instances"],
                sem_seg_results,
                self.combine_overlap_threshold,
                self.combine_stuff_area_limit,
                self.combine_instances_confidence_threshold,
                self.roi_heads.test_detections_per_img,
                self.sem_seg_head.num_classes
            )
            results["panoptic_seg"] = segments_info
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = batched_inputs["image"]
        images = self.normalizer(images)
        if self.input_format == "BGR": images = images[..., ::-1]
        image_shapes = batched_inputs["image_shape"]
        images = ImageList.from_tensors(
            images, image_shapes, self.neck.size_divisibility
        )
        return images


def combine_semantic_and_instance_outputs(
    detected_results,
    sem_seg,
    overlap_threshold,
    stuff_area_limit,
    instances_confidence_threshold,
    max_num_instances,
    num_stuff_classes,
):
    """
    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.
    Args:
        detected_instances: output of :func:`detector_postprocess`.
        sem_seg: semantic_results: an (N, H, W) tensor, each is the contiguous semantic
            category id
    Returns:
        dict containing fields:
            pano_seg (Tensor): of shape (height, width) where the values are ids for each segment.
    """
    result_fields = fields.ResultFields
    assert result_fields.masks in detected_results

    def combine_single_image(args):
        instance_dict, sem_seg = args
        panoptic_seg = tf.zeros_like(sem_seg, dtype=tf.int32)

        current_segment_id = 0
        segments_info = defaultdict(list)

        sorted_indices = tf.argsort(-instance_dict[result_fields.scores])

        # Add instances one-by-one, check for overlaps with existing ones
        for i in range(max_num_instances):
            idx = sorted_indices[i]
            is_valid = instance_dict[result_fields.is_valid][idx]
            score = instance_dict[result_fields.scores][idx]

            category_id = instance_dict[result_fields.classes][idx]

            instance_mask = instance_dict[result_fields.masks][idx]
            mask_area = tf.reduce_sum(tf.cast(instance_mask, tf.float32))

            instersect = (instance_mask > 0) & (panoptic_seg > 0)
            intersect_area = tf.reduce_sum(tf.cast(instersect, tf.float32))

            is_valid = is_valid & (score > instances_confidence_threshold) & (
                mask_area > 0) & (intersect_area / mask_area < overlap_threshold)

            mask = tf.cond(
                is_valid,
                lambda: (instance_mask > 0) & ~(panoptic_seg > 0),
                lambda: tf.zeros_like(panoptic_seg, dtype=tf.bool)
            )

            box = instance_dict[result_fields.boxes][idx]
            current_segment_id += 1
            panoptic_seg += current_segment_id * tf.cast(mask, tf.int32)

            segment_info = {
                "id": current_segment_id,
                "isthing": True,
                "score": score,
                "category_id": category_id,
                "instance_id": idx,
                "bbox": box,
                "is_valid": is_valid,
                "area": mask_area
            }
            for key in segment_info:
                segments_info[key].append(segment_info[key])

        # Add semantic results to remaining empty areas
        one_hot_sem_seg = tf.one_hot(
            sem_seg, num_stuff_classes, on_value=True, off_value=False, dtype=tf.bool
        )
        for semantic_label in range(num_stuff_classes):  # -1 to drop background
            if semantic_label == 0:  # 0 is a special "thing" class
                continue
            semantic_mask = one_hot_sem_seg[..., semantic_label] & ~(panoptic_seg > 0)
            mask_area = tf.reduce_sum(tf.cast(semantic_mask, tf.float32))

            def true_fn():
                mask = one_hot_sem_seg[..., semantic_label] & ~(panoptic_seg > 0)
                non_zero_indices = tf.where(mask)
                min_vals = tf.reduce_min(non_zero_indices, axis=0)
                max_vals = tf.reduce_max(non_zero_indices, axis=0)
                box = tf.concat([min_vals, max_vals], axis=0)
                return box, mask

            def false_fn():
                mask = tf.zeros_like(sem_seg, tf.bool)
                box = tf.constant([0, 0, 0, 0], tf.int64)
                return box, mask

            is_valid = mask_area > stuff_area_limit
            box, mask = tf.cond(is_valid, true_fn, false_fn)

            current_segment_id += 1
            panoptic_seg += current_segment_id * tf.cast(mask, tf.int32)

            segment_info = {
                "id": current_segment_id,
                "isthing": False,
                "score": 0.5,
                "category_id": semantic_label,
                "instance_id": 0,
                "bbox": tf.cast(box, tf.float32),
                "is_valid": is_valid,
                "area": mask_area
            }
            for key in segment_info:
                segments_info[key].append(segment_info[key])

        segments_info = dict(segments_info)
        for key in segments_info:
            segments_info[key] = tf.stack(segments_info[key])
        segments_info["panoptic_seg"] = panoptic_seg
        return segments_info

    dtype = {
        "id": tf.int32,
        "isthing": tf.bool,
        "score": tf.float32,
        "category_id": tf.int64,
        "instance_id": tf.int32,
        "bbox": tf.float32,
        "is_valid": tf.bool,
        "area": tf.float32,
        "panoptic_seg": tf.int32
    }
    segments_info = tf.map_fn(
        combine_single_image, [detected_results, sem_seg], dtype=dtype
    )

    return segments_info
