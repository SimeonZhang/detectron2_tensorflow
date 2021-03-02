"""Class for evaluating object detections with COCO metrics."""
import numpy as np
import json
import tensorflow as tf

from ..data import fields
from . import evaluator
from . import coco_tools
from . import visualization


class CocoDetectionEvaluator(evaluator.Evaluator):
    """Class to evaluate COCO detection metrics."""

    def __init__(
        self,
        category_index,
        include_metrics_per_category=False,
        all_metrics_per_category=False,
        max_examples_to_draw=10,
        min_visualization_score_thresh=0.2,
    ):
        """Constructor.

        Args:
            category_index: a dict containing COCO-like category information keyed
                by the 'id' field of each category.
            include_metrics_per_category: If True, include metrics for each category.
            all_metrics_per_category: Whether to include all the summary metrics for
                each category in per_category_ap. Be careful with setting it to true if
                you have more than handful of categories, because it will pollute
                your mldash.
        """
        super(CocoDetectionEvaluator, self).__init__(category_index)
        # _image_ids is a dictionary that maps unique image ids to Booleans which
        # indicate whether a corresponding detection has been added.
        self._image_ids = {}
        self._groundtruth_list = []
        self._detection_boxes_list = []
        self._summaries = []
        self._category_id_set = set(self._category_index.keys())
        self._annotation_id = 1
        self._metrics = None
        self._include_metrics_per_category = include_metrics_per_category
        self._all_metrics_per_category = all_metrics_per_category
        self._max_examples_to_draw = max_examples_to_draw
        self._min_visualization_score_thresh = min_visualization_score_thresh

    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self._image_ids.clear()
        self._groundtruth_list = []
        self._detection_boxes_list = []
        self._summaries = []

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """Adds groundtruth for a single image to be used for evaluation.

        If the image has already been added, a warning is logged, and groundtruth is
        ignored.

        Args:
        image_id: A unique string/integer identifier for the image.
        groundtruth_dict: A dictionary containing -
            InputFields.gt_boxes: float32 numpy array of shape
            [num_boxes, 4] containing `num_boxes` groundtruth boxes of the format
            [ymin, xmin, ymax, xmax] in absolute image coordinates.
            InputFields.gt_classes: integer numpy array of shape
            [num_boxes] containing 1-indexed groundtruth classes for the boxes.
            InputFields.gt_is_crowd (optional): integer numpy array of
            shape [num_boxes] containing iscrowd flag for groundtruth boxes.
        """
        if image_id in self._image_ids:
            tf.logging.warning(
                'Ignoring ground truth with image id %s since it was previously added', image_id)
            return

        groundtruth_is_crowd = groundtruth_dict.get(fields.InputFields.gt_is_crowd)
        # Drop groundtruth_is_crowd if empty tensor.
        if groundtruth_is_crowd is not None and not groundtruth_is_crowd.shape[0]:
            groundtruth_is_crowd = None

        groundtruth_boxes = groundtruth_dict[fields.InputFields.gt_boxes]
        groundtruth_classes = groundtruth_dict[fields.InputFields.gt_classes]
        groundtruth_masks = groundtruth_dict.get(fields.InputFields.gt_masks)

        if len(self._image_ids) < self._max_examples_to_draw:
            image = visualization.visualize_boxes_and_labels_on_image_array(
                image=groundtruth_dict[fields.InputFields.image],
                boxes=groundtruth_boxes,
                classes=groundtruth_classes,
                scores=np.ones_like(groundtruth_classes),
                category_index=self._category_index,
                instance_masks=groundtruth_masks,
                min_score_thresh=self._min_visualization_score_thresh
            )
            self._summaries.append(
                tf.Summary.Value(
                    tag="{}/Groudtruth/Detection".format(image_id),
                    image=tf.Summary.Image(
                        encoded_image_string=visualization.encode_image_array_as_png_str(image)))
            )

        self._groundtruth_list.extend(
            coco_tools.ExportSingleImageGroundtruthToCoco(
                image_id=image_id,
                next_annotation_id=self._annotation_id,
                category_id_set=self._category_id_set,
                groundtruth_boxes=groundtruth_boxes,
                groundtruth_classes=groundtruth_classes,
                groundtruth_masks=groundtruth_masks,
                groundtruth_is_crowd=groundtruth_is_crowd))

        self._annotation_id += groundtruth_dict[fields.InputFields.gt_boxes].shape[0]
        # Boolean to indicate whether a detection has been added for this image.
        self._image_ids[image_id] = False

    def add_single_predicted_image_info(self, image_id, result_dict):
        """Adds detections for a single image to be used for evaluation.

        If a detection has already been added for this image id, a warning is
        logged, and the detection is skipped.

        Args:
        image_id: A unique string/integer identifier for the image.
        result_dict: A dictionary containing -
            ResultFields.boxes: float32 numpy array of shape
            [num_boxes, 4] containing `num_boxes` detection boxes of the format
            [ymin, xmin, ymax, xmax] in absolute image coordinates.
            ResultFields.scores: float32 numpy array of shape
            [num_boxes] containing detection scores for the boxes.
            ResultFields.classes: integer numpy array of shape
            [num_boxes] containing 1-indexed detection classes for the boxes.

        Raises:
            ValueError: If groundtruth for the image_id is not available.
        """
        if image_id not in self._image_ids:
            raise ValueError('Missing groundtruth for image id: {}'.format(image_id))

        if self._image_ids[image_id]:
            tf.logging.warning(
                'Ignoring detection with image id %s since it was previously added', image_id)
            return

        assert "instances" in result_dict
        instance_result = result_dict["instances"]
        detection_boxes = instance_result[fields.ResultFields.boxes]
        detection_scores = instance_result[fields.ResultFields.scores]
        detection_classes = instance_result[fields.ResultFields.classes]

        if len(self._image_ids) <= self._max_examples_to_draw:
            image = visualization.visualize_boxes_and_labels_on_image_array(
                image=result_dict[fields.InputFields.image],
                boxes=detection_boxes,
                classes=detection_classes,
                scores=detection_scores,
                category_index=self._category_index,
                instance_masks=instance_result.get(fields.ResultFields.masks),
                min_score_thresh=self._min_visualization_score_thresh
            )
            self._summaries.append(
                tf.Summary.Value(
                    tag="{}/Prediction/Detection".format(image_id),
                    image=tf.Summary.Image(
                        encoded_image_string=visualization.encode_image_array_as_png_str(image)))
            )

        self._detection_boxes_list.extend(
            coco_tools.ExportSingleImageDetectionBoxesToCoco(
                image_id=image_id,
                category_id_set=self._category_id_set,
                detection_boxes=detection_boxes,
                detection_scores=detection_scores,
                detection_classes=detection_classes))
        self._image_ids[image_id] = True

    def dump_detections_to_json_file(self, json_output_path):
        """Saves the detections into json_output_path in the format used by MS COCO.

        Args:
        json_output_path: String containing the output file's path. It can be also
            None. In that case nothing will be written to the output file.
        """
        if json_output_path and json_output_path is not None:
            with tf.gfile.GFile(json_output_path, 'w') as fid:
                tf.logging.info('Dumping detections to output json file.')
                json.dump(obj=self._detection_boxes_list, fid=fid, indent=2)

    def evaluate(self):
        """Evaluates the detection boxes and returns a dictionary of coco metrics.

        Returns:
            A dictionary holding -

            1. summary_metrics:
            'DetectionBoxes_Precision/mAP': mean average precision over classes
                averaged over IOU thresholds ranging from .5 to .95 with .05
                increments.
            'DetectionBoxes_Precision/mAP@.50IOU': mean average precision at 50% IOU
            'DetectionBoxes_Precision/mAP@.75IOU': mean average precision at 75% IOU
            'DetectionBoxes_Precision/mAP (small)': mean average precision for small
                objects (area < 32^2 pixels).
            'DetectionBoxes_Precision/mAP (medium)': mean average precision for
                medium sized objects (32^2 pixels < area < 96^2 pixels).
            'DetectionBoxes_Precision/mAP (large)': mean average precision for large
                objects (96^2 pixels < area < 10000^2 pixels).
            'DetectionBoxes_Recall/AR@1': average recall with 1 detection.
            'DetectionBoxes_Recall/AR@10': average recall with 10 detections.
            'DetectionBoxes_Recall/AR@100': average recall with 100 detections.
            'DetectionBoxes_Recall/AR@100_small': average recall for small objects
                with 100.
            'DetectionBoxes_Recall/AR@100_medium': average recall for medium objects
                with 100.
            'DetectionBoxes_Recall/AR@100_large': average recall for large objects
                with 100 detections.

            2. per_category_ap: if include_metrics_per_category is True, category
            specific results with keys of the form:
            'PrecisionByCategory/mAP/category' (without the supercategory part if
            no supercategories exist). For backward compatibility
            'PerformanceByCategory' is included in the output regardless of
            all_metrics_per_category.
        """
        groundtruth_dict = {
            'annotations': self._groundtruth_list,
            'images': [{'id': image_id} for image_id in self._image_ids],
            'categories': list(self._category_index.values())
        }
        coco_wrapped_groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
        coco_wrapped_detections = coco_wrapped_groundtruth.LoadAnnotations(
            self._detection_boxes_list)
        box_evaluator = coco_tools.COCOEvalWrapper(
            coco_wrapped_groundtruth, coco_wrapped_detections, agnostic_mode=False)
        box_metrics, box_per_category_ap = box_evaluator.ComputeMetrics(
            include_metrics_per_category=self._include_metrics_per_category,
            all_metrics_per_category=self._all_metrics_per_category)
        box_metrics.update(box_per_category_ap)
        box_metrics = {'DetectionBoxes_' + key: value for key, value in iter(box_metrics.items())}
        for key, value in box_metrics.items():
            self._summaries.append(
                tf.Summary.Value(tag=key, simple_value=value)
            )
        return box_metrics, self._summaries


def _check_mask_type_and_value(array_name, masks):
    """Checks whether mask dtype is uint8 and the values are either 0 or 1."""
    if masks.dtype != np.uint8:
        raise ValueError('{} must be of type np.uint8. Found {}.'.format(
            array_name, masks.dtype))
    if np.any(np.logical_and(masks != 0, masks != 1)):
        raise ValueError('{} elements can only be either 0 or 1.'.format(
            array_name))


class CocoMaskEvaluator(evaluator.Evaluator):
    """Class to evaluate COCO detection metrics."""

    def __init__(
        self,
        category_index,
        include_metrics_per_category=False,
        all_metrics_per_category=False,
        max_examples_to_draw=10,
        min_visualization_score_thresh=0.2,
    ):
        """Constructor.

        Args:
        categories: A list of dicts, each of which has the following keys -
            'id': (required) an integer id uniquely identifying this category.
            'name': (required) string representing category name e.g., 'cat', 'dog'.
        include_metrics_per_category: If True, include metrics for each category.
        """
        super(CocoMaskEvaluator, self).__init__(category_index)
        self._image_id_to_mask_shape_map = {}
        self._image_ids_with_detections = set([])
        self._groundtruth_list = []
        self._detection_masks_list = []
        self._summaries = []
        self._category_id_set = set(self._category_index.keys())
        self._annotation_id = 1
        self._include_metrics_per_category = include_metrics_per_category
        self._all_metrics_per_category = all_metrics_per_category
        self._max_examples_to_draw = max_examples_to_draw
        self._min_visualization_score_thresh = min_visualization_score_thresh

    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self._image_id_to_mask_shape_map.clear()
        self._image_ids_with_detections.clear()
        self._groundtruth_list = []
        self._detection_masks_list = []
        self._summaries = []

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """Adds groundtruth for a single image to be used for evaluation.

        If the image has already been added, a warning is logged, and groundtruth is
        ignored.

        Args:
        image_id: A unique string/integer identifier for the image.
        groundtruth_dict: A dictionary containing -
            InputFields.groundtruth_boxes: float32 numpy array of shape
            [num_boxes, 4] containing `num_boxes` groundtruth boxes of the format
            [ymin, xmin, ymax, xmax] in absolute image coordinates.
            InputFields.groundtruth_classes: integer numpy array of shape
            [num_boxes] containing 1-indexed groundtruth classes for the boxes.
            InputFields.groundtruth_instance_masks: uint8 numpy array of shape
            [num_boxes, image_height, image_width] containing groundtruth masks
            corresponding to the boxes. The elements of the array must be in
            {0, 1}.
        """
        if image_id in self._image_id_to_mask_shape_map:
            tf.logging.warning('Ignoring ground truth with image id %s since it was '
                               'previously added', image_id)
            return

        groundtruth_boxes = groundtruth_dict[fields.InputFields.gt_boxes]
        groundtruth_classes = groundtruth_dict[fields.InputFields.gt_classes]
        groundtruth_masks = groundtruth_dict[fields.InputFields.gt_masks]
        _check_mask_type_and_value(fields.InputFields.gt_masks, groundtruth_masks)

        if len(self._image_id_to_mask_shape_map) < self._max_examples_to_draw:
            image = visualization.visualize_boxes_and_labels_on_image_array(
                image=groundtruth_dict[fields.InputFields.image],
                boxes=groundtruth_boxes,
                classes=groundtruth_classes,
                scores=np.ones_like(groundtruth_classes),
                category_index=self._category_index,
                instance_masks=groundtruth_masks,
                min_score_thresh=self._min_visualization_score_thresh
            )
            self._summaries.append(
                tf.Summary.Value(
                    tag="{}/Groudtruth/Detection".format(image_id),
                    image=tf.Summary.Image(
                        encoded_image_string=visualization.encode_image_array_as_png_str(image)))
            )

        self._groundtruth_list.extend(
            coco_tools.
            ExportSingleImageGroundtruthToCoco(
                image_id=image_id,
                next_annotation_id=self._annotation_id,
                category_id_set=self._category_id_set,
                groundtruth_boxes=groundtruth_boxes,
                groundtruth_classes=groundtruth_classes,
                groundtruth_masks=groundtruth_masks))
        self._annotation_id += groundtruth_boxes.shape[0]
        self._image_id_to_mask_shape_map[image_id] = groundtruth_dict[
            fields.InputFields.gt_masks].shape

    def add_single_predicted_image_info(self, image_id, result_dict):
        """Adds detections for a single image to be used for evaluation.

        If a detection has already been added for this image id, a warning is
        logged, and the detection is skipped.

        Args:
            image_id: A unique string/integer identifier for the image.
            result_dict: A dictionary containing -
                fields.ResultFields.scores: float32 numpy array of shape
                [num_boxes] containing detection scores for the boxes.
                fields.ResultFields.classes: integer numpy array of shape
                [num_boxes] containing 1-indexed detection classes for the boxes.
                fields.ResultFields.masks: optional uint8 numpy array of
                shape [num_boxes, image_height, image_width] containing instance
                masks corresponding to the boxes. The elements of the array must be
                in {0, 1}.

        Raises:
            ValueError: If groundtruth for the image_id is not available or if
                spatial shapes of groundtruth_instance_masks and detection_masks are
                incompatible.
        """
        if image_id not in self._image_id_to_mask_shape_map:
            raise ValueError('Missing groundtruth for image id: {}'.format(image_id))

        if image_id in self._image_ids_with_detections:
            tf.logging.warning('Ignoring detection with image id %s since it was '
                               'previously added', image_id)
            return

        assert "instances" in result_dict, result_dict.keys()
        instance_result = result_dict["instances"]
        groundtruth_masks_shape = self._image_id_to_mask_shape_map[image_id]
        detection_masks = instance_result[fields.ResultFields.masks]
        if groundtruth_masks_shape[1:] != detection_masks.shape[1:]:
            raise ValueError('Spatial shape of groundtruth masks and detection masks '
                             'are incompatible: {} vs {}'.format(
                                 groundtruth_masks_shape,
                                 detection_masks.shape))
        _check_mask_type_and_value(fields.ResultFields.masks, detection_masks)

        detection_boxes = instance_result[fields.ResultFields.boxes]
        detection_scores = instance_result[fields.ResultFields.scores]
        detection_classes = instance_result[fields.ResultFields.classes]

        if len(self._image_ids_with_detections) < self._max_examples_to_draw:
            image = visualization.visualize_boxes_and_labels_on_image_array(
                image=result_dict[fields.InputFields.image],
                boxes=detection_boxes,
                classes=detection_classes,
                scores=detection_scores,
                category_index=self._category_index,
                instance_masks=detection_masks,
                min_score_thresh=self._min_visualization_score_thresh
            )
            self._summaries.append(
                tf.Summary.Value(
                    tag="{}/Prediction/Detection".format(image_id),
                    image=tf.Summary.Image(
                        encoded_image_string=visualization.encode_image_array_as_png_str(image)))
            )
        self._detection_masks_list.extend(
            coco_tools.ExportSingleImageDetectionMasksToCoco(
                image_id=image_id,
                category_id_set=self._category_id_set,
                detection_masks=detection_masks,
                detection_scores=detection_scores,
                detection_classes=detection_classes))
        self._image_ids_with_detections.update([image_id])

    def dump_detections_to_json_file(self, json_output_path):
        """Saves the detections into json_output_path in the format used by MS COCO.

        Args:
        json_output_path: String containing the output file's path. It can be also
            None. In that case nothing will be written to the output file.
        """
        if json_output_path and json_output_path is not None:
            tf.logging.info('Dumping detections to output json file.')
            with tf.gfile.GFile(json_output_path, 'w') as fid:
                json.dump(obj=self._detection_masks_list, fid=fid, indent=2)

    def evaluate(self):
        """Evaluates the detection masks and returns a dictionary of coco metrics.

        Returns:
            A dictionary holding -

            1. summary_metrics:
            'DetectionMasks_Precision/mAP': mean average precision over classes
                averaged over IOU thresholds ranging from .5 to .95 with .05 increments.
            'DetectionMasks_Precision/mAP@.50IOU': mean average precision at 50% IOU.
            'DetectionMasks_Precision/mAP@.75IOU': mean average precision at 75% IOU.
            'DetectionMasks_Precision/mAP (small)': mean average precision for small
                objects (area < 32^2 pixels).
            'DetectionMasks_Precision/mAP (medium)': mean average precision for medium
                sized objects (32^2 pixels < area < 96^2 pixels).
            'DetectionMasks_Precision/mAP (large)': mean average precision for large
                objects (96^2 pixels < area < 10000^2 pixels).
            'DetectionMasks_Recall/AR@1': average recall with 1 detection.
            'DetectionMasks_Recall/AR@10': average recall with 10 detections.
            'DetectionMasks_Recall/AR@100': average recall with 100 detections.
            'DetectionMasks_Recall/AR@100_small': average recall for small objects
                with 100 detections.
            'DetectionMasks_Recall/AR@100_medium': average recall for medium objects
                with 100 detections.
            'DetectionMasks_Recall/AR@100_large': average recall for large objects
                with 100 detections.

            2. per_category_ap: if include_metrics_per_category is True, category
            specific results with keys of the form:
            'PrecisionByCategory/mAP/category' (without the supercategory part if
            no supercategories exist). For backward compatibility
            'PerformanceByCategory' is included in the output regardless of
            all_metrics_per_category.
        """
        groundtruth_dict = {
            'annotations': self._groundtruth_list,
            'images': [{
                'id': image_id, 'height': shape[1], 'width': shape[2]}
                for image_id, shape in self._image_id_to_mask_shape_map.items()],
            'categories': list(self._category_index.values())
        }
        coco_wrapped_groundtruth = coco_tools.COCOWrapper(
            groundtruth_dict, detection_type='segmentation')
        coco_wrapped_detection_masks = coco_wrapped_groundtruth.LoadAnnotations(
            self._detection_masks_list)
        mask_evaluator = coco_tools.COCOEvalWrapper(
            coco_wrapped_groundtruth, coco_wrapped_detection_masks,
            agnostic_mode=False, iou_type='segm')
        mask_metrics, mask_per_category_ap = mask_evaluator.ComputeMetrics(
            include_metrics_per_category=self._include_metrics_per_category,
            all_metrics_per_category=self._all_metrics_per_category)
        mask_metrics.update(mask_per_category_ap)
        mask_metrics = {'DetectionMasks_' + key: value
                        for key, value in mask_metrics.items()}
        for key, value in mask_metrics.items():
            self._summaries.append(
                tf.Summary.Value(tag=key, simple_value=value)
            )
        return mask_metrics, self._summaries
