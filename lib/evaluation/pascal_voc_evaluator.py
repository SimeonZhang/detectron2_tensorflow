import collections
import numpy as np
import tensorflow as tf

from ..data import fields
from . import metrics
from . import per_image_evaluation
from . import evaluator
from . import visualization

__all__ = [
    "PascalDetectionEvaluator",
    "WeightedPascalDetectionEvaluator",
    "PascalInstanceSegmentationEvaluator",
    "WeightedPascalInstanceSegmentationEvaluator",
]


class ObjectDetectionEvaluator(evaluator.Evaluator):
    """A class to evaluate detections."""

    def __init__(
        self,
        category_index,
        matching_iou_threshold=0.5,
        evaluate_corlocs=False,
        evaluate_precision_recall=False,
        metric_prefix=None,
        use_weighted_mean_ap=False,
        evaluate_masks=False,
        group_of_weight=0.0,
        max_examples_to_draw=10,
        min_visualization_score_thresh=0.2,
    ):
        """Constructor.
        Args:
            category_index: a dict containing COCO-like category information keyed
                by the 'id' field of each category.
            matching_iou_threshold: IOU threshold to use for matching groundtruth
                boxes to detection boxes.
            evaluate_corlocs: (optional) boolean which determines if corloc scores
                are to be returned or not.
            evaluate_precision_recall: (optional) boolean which determines if
                precision and recall values are to be returned or not.
            metric_prefix: (optional) string prefix for metric name; if None, no
                prefix is used.
            use_weighted_mean_ap: (optional) boolean which determines if the mean
                average precision is computed directly from the scores and tp_fp_labels
                of all classes.
            evaluate_masks: If False, evaluation will be performed based on boxes.
                If True, mask evaluation will be performed instead.
            group_of_weight: Weight of group-of boxes.If set to 0, detections of the
                correct class within a group-of box are ignored. If weight is > 0, then
                if at least one detection falls within a group-of box with
                matching_iou_threshold, weight group_of_weight is added to true
                positives. Consequently, if no detection falls within a group-of box,
                weight group_of_weight is added to false negatives.
        Raises:
            ValueError: If the category ids are not 1-indexed.
        """
        super(ObjectDetectionEvaluator, self).__init__(category_index)
        self._num_classes = max(category_index.keys()) + 1
        self._matching_iou_threshold = matching_iou_threshold
        self._use_weighted_mean_ap = use_weighted_mean_ap
        self._evaluate_masks = evaluate_masks
        self._group_of_weight = group_of_weight
        self._evaluation = ObjectDetectionEvaluation(
            num_groundtruth_classes=self._num_classes,
            matching_iou_threshold=self._matching_iou_threshold,
            use_weighted_mean_ap=self._use_weighted_mean_ap,
            group_of_weight=self._group_of_weight)
        self._image_ids_predicted = {}
        self._summaries = []
        self._evaluate_corlocs = evaluate_corlocs
        self._evaluate_precision_recall = evaluate_precision_recall
        self._max_examples_to_draw = max_examples_to_draw
        self._min_visualization_score_thresh = min_visualization_score_thresh
        self._metric_prefix = (metric_prefix + '_') if metric_prefix else ''
        self._expected_keys = set([
            fields.InputFields.key,
            fields.InputFields.gt_boxes,
            fields.InputFields.gt_classes,
            fields.InputFields.gt_difficult,
            fields.InputFields.gt_masks,
            fields.ResultFields.boxes,
            fields.ResultFields.scores,
            fields.ResultFields.classes,
            fields.ResultFields.masks
        ])
        self._build_metric_names()

    def _build_metric_names(self):
        """Builds a list with metric names."""

        self._metric_names = [
            self._metric_prefix + 'Precision/mAP@{}IOU'.format(
                self._matching_iou_threshold)
        ]
        if self._evaluate_corlocs:
            self._metric_names.append(
                self._metric_prefix +
                'Precision/meanCorLoc@{}IOU'.format(self._matching_iou_threshold)
            )

        for idx in range(self._num_classes):
            if idx in self._category_index:
                category_name = self._category_index[idx]['name']
                self._metric_names.append(
                    self._metric_prefix + 'PerformanceByCategory/AP@{}IOU/{}'.format(
                        self._matching_iou_threshold, category_name)
                )
                if self._evaluate_corlocs:
                    self._metric_names.append(
                        self._metric_prefix + 'PerformanceByCategory/CorLoc@{}IOU/{}'.format(
                            self._matching_iou_threshold, category_name)
                    )

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """Adds groundtruth for a single image to be used for evaluation.
        Args:
            image_id: A unique string/integer identifier for the image.
            groundtruth_dict: A dictionary containing -
                fields.InputFields.gt_boxes: float32 numpy array
                of shape [num_boxes, 4] containing `num_boxes` groundtruth boxes of
                the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
                fields.InputFields.gt_classes: integer numpy array of shape [num_boxes]
                containing 1-indexed groundtruth classes for the boxes.
                fields.InputFields.gt_difficult: Optional length num_boxes numpy boolean
                array denoting whether a ground truth box is a difficult instance or not.
                This field is optional to support the case that no boxes are difficult.
                fields.InputFields.gt_masks: Optional numpy array of shape
                [num_boxes, height, width] with values in {0, 1}.
        Raises:
            ValueError: On adding groundtruth for an image more than once.
                Will also raise error if evaluate_masks and instance masks are not in
                groundtruth dictionary.
        """
        if image_id in self._image_ids_predicted:
            tf.logging.warning('Image with id {} already added.'.format(image_id))
            return

        groundtruth_boxes = groundtruth_dict[fields.InputFields.gt_boxes]
        groundtruth_classes = groundtruth_dict[fields.InputFields.gt_classes]
        groundtruth_difficult = groundtruth_dict.get(fields.InputFields.gt_difficult)
        groundtruth_group_of = groundtruth_dict.get(fields.InputFields.gt_is_crowd)
        if not len(self._image_ids_predicted) % 1000:
            tf.logging.warning(
                'image %s does not have gt difficult flag specified', image_id)
        groundtruth_masks = None
        if self._evaluate_masks:
            if fields.InputFields.gt_masks not in groundtruth_dict:
                raise ValueError('Instance masks not in groundtruth dictionary.')
            groundtruth_masks = groundtruth_dict[fields.InputFields.gt_masks]

        if len(self._image_ids_predicted) < self._max_examples_to_draw:
            image = visualization.visualize_boxes_and_labels_on_image_array(
                image=groundtruth_dict[fields.InputFields.image],
                boxes=groundtruth_boxes,
                classes=groundtruth_classes,
                scores=None,
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

        self._evaluation.add_single_ground_truth_image_info(
            image_key=image_id,
            groundtruth_boxes=groundtruth_boxes,
            groundtruth_class_labels=groundtruth_classes,
            groundtruth_is_difficult_list=groundtruth_difficult,
            groundtruth_is_group_of_list=groundtruth_group_of,
            groundtruth_masks=groundtruth_masks)
        self._image_ids_predicted[image_id] = False

    def add_single_predicted_image_info(self, image_id, result_dict):
        """Adds detections for a single image to be used for evaluation.
        Args:
        image_id: A unique string/integer identifier for the image.
        result_dict: A dictionary containing -
            fields.ResultFields.classes: float32 numpy array of shape [num_boxes, 4]
            containing `num_boxes` detection boxes of the format
            [ymin, xmin, ymax, xmax] in absolute image coordinates.
            fields.DetectionFields.detection_scores: float32 numpy
            array of shape [num_boxes] containing detection scores for the boxes.
            fields.DetectionFields.detection_classes: integer numpy
            array of shape [num_boxes] containing 1-indexed detection classes for
            the boxes.
            fields.ResultFields.masks: uint8 numpy
            array of shape [num_boxes, height, width] containing `num_boxes` masks
            of values ranging between 0 and 1.
        Raises:
        ValueError: If detection masks are not in detections dictionary.
        """
        if image_id not in self._image_ids_predicted:
            tf.logging.warning('Ignoring detection with image id %s since there was '
                               'no groundtruth added', image_id)
            return

        if self._image_ids_predicted[image_id]:
            tf.logging.warning('Ignoring detection with image id %s since it was '
                               'previously added', image_id)
            return

        assert "instances" in result_dict, result_dict.keys()
        instance_result = result_dict["instances"]
        detection_boxes = instance_result[fields.ResultFields.boxes]
        detection_classes = instance_result[fields.ResultFields.classes]
        detection_scores = instance_result[fields.ResultFields.scores]
        detection_masks = None
        if self._evaluate_masks:
            if fields.ResultFields.masks not in instance_result:
                raise ValueError('Masks not in results dictionary.')
            detection_masks = instance_result[fields.ResultFields.masks]

        if len(self._image_ids_predicted) <= self._max_examples_to_draw:
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

        self._evaluation.add_single_predicted_image_info(
            image_key=image_id,
            detected_boxes=detection_boxes,
            detected_scores=detection_scores,
            detected_class_labels=detection_classes,
            detected_masks=detection_masks)

    def evaluate(self):
        """Compute evaluation result.
        Returns:
        A dictionary of metrics with the following fields -
        1. summary_metrics:
            '<prefix if not empty>_Precision/mAP@<matching_iou_threshold>IOU': mean
            average precision at the specified IOU threshold.
        2. per_category_ap: category specific results with keys of the form
            '<prefix if not empty>_PerformanceByCategory/
            mAP@<matching_iou_threshold>IOU/category'.
        """
        (per_class_ap, mean_ap, per_class_precision, per_class_recall,
            per_class_corloc, mean_corloc) = self._evaluation.evaluate()
        pascal_metrics = {self._metric_names[0]: mean_ap}
        if self._evaluate_corlocs:
            pascal_metrics[self._metric_names[1]] = mean_corloc
        for idx in range(per_class_ap.size):
            if idx in self._category_index:
                category_name = self._category_index[idx]['name']
                display_name = (
                    self._metric_prefix + 'PerformanceByCategory/AP@{}IOU/{}'.format(
                        self._matching_iou_threshold, category_name))
                pascal_metrics[display_name] = per_class_ap[idx]

                # Optionally add precision and recall values
                if self._evaluate_precision_recall:
                    display_name = (
                        self._metric_prefix +
                        'PerformanceByCategory/Precision@{}IOU/{}'.format(
                            self._matching_iou_threshold, category_name))
                    pascal_metrics[display_name] = per_class_precision[idx]
                    display_name = (
                        self._metric_prefix +
                        'PerformanceByCategory/Recall@{}IOU/{}'.format(
                            self._matching_iou_threshold, category_name))
                    pascal_metrics[display_name] = per_class_recall[idx]

                # Optionally add CorLoc metrics.classes
                if self._evaluate_corlocs:
                    display_name = (
                        self._metric_prefix + 'PerformanceByCategory/CorLoc@{}IOU/{}'
                        .format(self._matching_iou_threshold, category_name))
                    pascal_metrics[display_name] = per_class_corloc[idx]
        for key, value in pascal_metrics.items():
            self._summaries.append(
                tf.Summary.Value(tag=key, simple_value=value)
            )
        return pascal_metrics, self._summaries

    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self._evaluation = ObjectDetectionEvaluation(
            num_groundtruth_classes=self._num_classes,
            matching_iou_threshold=self._matching_iou_threshold,
            use_weighted_mean_ap=self._use_weighted_mean_ap)
        self._image_ids_predicted = {}
        self._summaries = []


ObjectDetectionEvalMetrics = collections.namedtuple(
    'ObjectDetectionEvalMetrics', [
        'average_precisions', 'mean_ap', 'precisions', 'recalls', 'corlocs',
        'mean_corloc'
    ])


class ObjectDetectionEvaluation(object):
    """Internal implementation of Pascal object detection metrics."""

    def __init__(
        self,
        num_groundtruth_classes,
        matching_iou_threshold=0.5,
        nms_iou_threshold=1.0,
        nms_max_output_boxes=10000,
        use_weighted_mean_ap=False,
        label_id_offset=0,
        group_of_weight=0.0,
        per_image_eval_class=per_image_evaluation.PerImageDetectionEvaluation
    ):
        """Constructor.
        Args:
        num_groundtruth_classes: Number of ground-truth classes.
        matching_iou_threshold: IOU threshold used for matching detected boxes
            to ground-truth boxes.
        nms_iou_threshold: IOU threshold used for non-maximum suppression.
        nms_max_output_boxes: Maximum number of boxes returned by non-maximum
            suppression.
        use_weighted_mean_ap: (optional) boolean which determines if the mean
            average precision is computed directly from the scores and tp_fp_labels
            of all classes.
        group_of_weight: Weight of group-of boxes.If set to 0, detections of the
            correct class within a group-of box are ignored. If weight is > 0, then
            if at least one detection falls within a group-of box with
            matching_iou_threshold, weight group_of_weight is added to true
            positives. Consequently, if no detection falls within a group-of box,
            weight group_of_weight is added to false negatives.
        per_image_eval_class: The class that contains functions for computing
            per image metrics.
        Raises:
            ValueError: if num_groundtruth_classes is smaller than 1.
        """
        if num_groundtruth_classes < 1:
            raise ValueError('Need at least 1 groundtruth class for evaluation.')

        self.per_image_eval = per_image_eval_class(
            num_groundtruth_classes=num_groundtruth_classes,
            matching_iou_threshold=matching_iou_threshold,
            nms_iou_threshold=nms_iou_threshold,
            nms_max_output_boxes=nms_max_output_boxes,
            group_of_weight=group_of_weight)
        self.label_id_offset = label_id_offset
        self.group_of_weight = group_of_weight
        self.num_class = num_groundtruth_classes
        self.use_weighted_mean_ap = use_weighted_mean_ap

        self.groundtruth_boxes = {}
        self.groundtruth_class_labels = {}
        self.groundtruth_masks = {}
        self.groundtruth_is_difficult_list = {}
        self.groundtruth_is_group_of_list = {}
        self.num_gt_instances_per_class = np.zeros(self.num_class, dtype=float)
        self.num_gt_imgs_per_class = np.zeros(self.num_class, dtype=int)

        self._initialize_detections()

    def _initialize_detections(self):
        """Initializes internal data structures."""
        self.detection_keys = set()
        self.scores_per_class = [[] for _ in range(self.num_class)]
        self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
        self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
        self.average_precision_per_class = np.empty(self.num_class, dtype=float)
        self.average_precision_per_class.fill(np.nan)
        self.precisions_per_class = [np.nan] * self.num_class
        self.recalls_per_class = [np.nan] * self.num_class

        self.corloc_per_class = np.ones(self.num_class, dtype=float)

    def clear_detections(self):
        self._initialize_detections()

    def add_single_ground_truth_image_info(
        self,
        image_key,
        groundtruth_boxes,
        groundtruth_class_labels,
        groundtruth_is_difficult_list=None,
        groundtruth_is_group_of_list=None,
        groundtruth_masks=None
    ):
        """Adds groundtruth for a single image to be used for evaluation.
        Args:
        image_key: A unique string/integer identifier for the image.
        groundtruth_boxes: float32 numpy array of shape [num_boxes, 4]
            containing `num_boxes` groundtruth boxes of the format
            [ymin, xmin, ymax, xmax] in absolute image coordinates.
        groundtruth_class_labels: integer numpy array of shape [num_boxes]
            containing 0-indexed groundtruth classes for the boxes.
        groundtruth_is_difficult_list: A length M numpy boolean array denoting
            whether a ground truth box is a difficult instance or not. To support
            the case that no boxes are difficult, it is by default set as None.
        groundtruth_is_group_of_list: A length M numpy boolean array denoting
            whether a ground truth box is a group-of box or not. To support
            the case that no boxes are groups-of, it is by default set as None.
        groundtruth_masks: uint8 numpy array of shape
            [num_boxes, height, width] containing `num_boxes` groundtruth masks.
            The mask values range from 0 to 1.
        """
        if image_key in self.groundtruth_boxes:
            tf.logging.warning(
                'image %s has already been added to the ground truth database.',
                image_key)
            return
        self.groundtruth_boxes[image_key] = groundtruth_boxes
        self.groundtruth_class_labels[image_key] = groundtruth_class_labels
        self.groundtruth_masks[image_key] = groundtruth_masks
        if groundtruth_is_difficult_list is None:
            num_boxes = groundtruth_boxes.shape[0]
            groundtruth_is_difficult_list = np.zeros(num_boxes, dtype=bool)
        self.groundtruth_is_difficult_list[
            image_key] = groundtruth_is_difficult_list.astype(dtype=bool)
        if groundtruth_is_group_of_list is None:
            num_boxes = groundtruth_boxes.shape[0]
            groundtruth_is_group_of_list = np.zeros(num_boxes, dtype=bool)
        self.groundtruth_is_group_of_list[
            image_key] = groundtruth_is_group_of_list.astype(dtype=bool)

        self._update_ground_truth_statistics(
            groundtruth_class_labels,
            groundtruth_is_difficult_list.astype(dtype=bool),
            groundtruth_is_group_of_list.astype(dtype=bool))

    def add_single_predicted_image_info(
        self,
        image_key,
        detected_boxes,
        detected_scores,
        detected_class_labels,
        detected_masks=None
    ):
        """Adds detections for a single image to be used for evaluation.
        Args:
        image_key: A unique string/integer identifier for the image.
        detected_boxes: float32 numpy array of shape [num_boxes, 4]
            containing `num_boxes` detection boxes of the format
            [ymin, xmin, ymax, xmax] in absolute image coordinates.
        detected_scores: float32 numpy array of shape [num_boxes] containing
            detection scores for the boxes.
        detected_class_labels: integer numpy array of shape [num_boxes] containing
            0-indexed detection classes for the boxes.
        detected_masks: np.uint8 numpy array of shape [num_boxes, height, width]
            containing `num_boxes` detection masks with values ranging
            between 0 and 1.
        Raises:
        ValueError: if the number of boxes, scores and class labels differ in
            length.
        """
        if (len(detected_boxes) != len(detected_scores) or
                len(detected_boxes) != len(detected_class_labels)):
            raise ValueError('detected_boxes, detected_scores and '
                             'detected_class_labels should all have same lengths. Got'
                             '[%d, %d, %d]' % len(detected_boxes),
                             len(detected_scores), len(detected_class_labels))

        if image_key in self.detection_keys:
            tf.logging.warning(
                'image %s has already been added to the detection result database',
                image_key)
            return

        self.detection_keys.add(image_key)
        if image_key in self.groundtruth_boxes:
            groundtruth_boxes = self.groundtruth_boxes[image_key]
            groundtruth_class_labels = self.groundtruth_class_labels[image_key]
            # Masks are popped instead of look up. The reason is that we do not want
            # to keep all masks in memory which can cause memory overflow.
            groundtruth_masks = self.groundtruth_masks.pop(image_key)
            groundtruth_is_difficult_list = self.groundtruth_is_difficult_list[image_key]
            groundtruth_is_group_of_list = self.groundtruth_is_group_of_list[image_key]
        else:
            groundtruth_boxes = np.empty(shape=[0, 4], dtype=float)
            groundtruth_class_labels = np.array([], dtype=int)
            if detected_masks is None:
                groundtruth_masks = None
            else:
                groundtruth_masks = np.empty(shape=[0, 1, 1], dtype=float)
            groundtruth_is_difficult_list = np.array([], dtype=bool)
            groundtruth_is_group_of_list = np.array([], dtype=bool)
        scores, tp_fp_labels, is_class_correctly_detected_in_image = (
            self.per_image_eval.compute_object_detection_metrics(
                detected_boxes=detected_boxes,
                detected_scores=detected_scores,
                detected_class_labels=detected_class_labels,
                groundtruth_boxes=groundtruth_boxes,
                groundtruth_class_labels=groundtruth_class_labels,
                groundtruth_is_difficult_list=groundtruth_is_difficult_list,
                groundtruth_is_group_of_list=groundtruth_is_group_of_list,
                detected_masks=detected_masks,
                groundtruth_masks=groundtruth_masks))

        for i in range(self.num_class):
            if scores[i].shape[0] > 0:
                self.scores_per_class[i].append(scores[i])
                self.tp_fp_labels_per_class[i].append(tp_fp_labels[i])
        (self.num_images_correctly_detected_per_class
         ) += is_class_correctly_detected_in_image

    def _update_ground_truth_statistics(self, groundtruth_class_labels,
                                        groundtruth_is_difficult_list,
                                        groundtruth_is_group_of_list):
        """Update grouth truth statitistics.
        1. Difficult boxes are ignored when counting the number of ground truth
        instances as done in Pascal VOC devkit.
        2. Difficult boxes are treated as normal boxes when computing CorLoc related
        statitistics.
        Args:
        groundtruth_class_labels: An integer numpy array of length M,
            representing M class labels of object instances in ground truth
        groundtruth_is_difficult_list: A boolean numpy array of length M denoting
            whether a ground truth box is a difficult instance or not
        groundtruth_is_group_of_list: A boolean numpy array of length M denoting
            whether a ground truth box is a group-of box or not
        """
        for class_index in range(self.num_class):
            num_gt_instances = np.sum(groundtruth_class_labels[
                ~groundtruth_is_difficult_list & ~groundtruth_is_group_of_list] == class_index)
            num_groupof_gt_instances = self.group_of_weight * np.sum(
                groundtruth_class_labels[groundtruth_is_group_of_list] == class_index)
            self.num_gt_instances_per_class[
                class_index] += num_gt_instances + num_groupof_gt_instances
            if np.any(groundtruth_class_labels == class_index):
                self.num_gt_imgs_per_class[class_index] += 1

    def evaluate(self):
        """Compute evaluation result.
        Returns:
            A named tuple with the following fields -
                average_precision: float numpy array of average precision for
                    each class.
                mean_ap: mean average precision of all classes, float scalar
                precisions: List of precisions, each precision is a float numpy
                    array
                recalls: List of recalls, each recall is a float numpy array
                corloc: numpy float array
                mean_corloc: Mean CorLoc score for each class, float scalar
        """
        if (self.num_gt_instances_per_class == 0).any():
            tf.logging.warning(
                'The following classes have no ground truth examples: %s',
                np.squeeze(np.argwhere(self.num_gt_instances_per_class == 0)) +
                self.label_id_offset)

        if self.use_weighted_mean_ap:
            all_scores = np.array([], dtype=float)
            all_tp_fp_labels = np.array([], dtype=bool)
        for class_index in range(self.num_class):
            if self.num_gt_instances_per_class[class_index] == 0:
                continue
            if not self.scores_per_class[class_index]:
                scores = np.array([], dtype=float)
                tp_fp_labels = np.array([], dtype=float)
            else:
                scores = np.concatenate(self.scores_per_class[class_index])
                tp_fp_labels = np.concatenate(self.tp_fp_labels_per_class[class_index])
            if self.use_weighted_mean_ap:
                all_scores = np.append(all_scores, scores)
                all_tp_fp_labels = np.append(all_tp_fp_labels, tp_fp_labels)
            precision, recall = metrics.compute_precision_recall(
                scores, tp_fp_labels, self.num_gt_instances_per_class[class_index])

            self.precisions_per_class[class_index] = precision
            self.recalls_per_class[class_index] = recall
            average_precision = metrics.compute_average_precision(precision, recall)
            self.average_precision_per_class[class_index] = average_precision

        self.corloc_per_class = metrics.compute_cor_loc(
            self.num_gt_imgs_per_class,
            self.num_images_correctly_detected_per_class)

        if self.use_weighted_mean_ap:
            num_gt_instances = np.sum(self.num_gt_instances_per_class)
            precision, recall = metrics.compute_precision_recall(
                all_scores, all_tp_fp_labels, num_gt_instances)
            mean_ap = metrics.compute_average_precision(precision, recall)
        else:
            mean_ap = np.nanmean(self.average_precision_per_class)
        mean_corloc = np.nanmean(self.corloc_per_class)
        return ObjectDetectionEvalMetrics(
            self.average_precision_per_class, mean_ap, self.precisions_per_class,
            self.recalls_per_class, self.corloc_per_class, mean_corloc)


class PascalDetectionEvaluator(ObjectDetectionEvaluator):
    """A class to evaluate detections using PASCAL metrics."""

    def __init__(
        self,
        category_index,
        matching_iou_threshold=0.5,
        max_examples_to_draw=10,
        min_visualization_score_thresh=0.2
    ):
        super(PascalDetectionEvaluator, self).__init__(
            category_index,
            matching_iou_threshold=matching_iou_threshold,
            evaluate_corlocs=False,
            metric_prefix='PascalBoxes',
            use_weighted_mean_ap=False)


class WeightedPascalDetectionEvaluator(ObjectDetectionEvaluator):
    """A class to evaluate detections using weighted PASCAL metrics.
    Weighted PASCAL metrics computes the mean average precision as the average
    precision given the scores and tp_fp_labels of all classes. In comparison,
    PASCAL metrics computes the mean average precision as the mean of the
    per-class average precisions.
    This definition is very similar to the mean of the per-class average
    precisions weighted by class frequency. However, they are typically not the
    same as the average precision is not a linear function of the scores and
    tp_fp_labels.
    """

    def __init__(
        self,
        category_index,
        matching_iou_threshold=0.5,
        max_examples_to_draw=10,
        min_visualization_score_thresh=0.2
    ):
        super(WeightedPascalDetectionEvaluator, self).__init__(
            category_index,
            matching_iou_threshold=matching_iou_threshold,
            evaluate_corlocs=False,
            metric_prefix='WeightedPascalBoxes',
            use_weighted_mean_ap=True)


class PascalInstanceSegmentationEvaluator(ObjectDetectionEvaluator):
    """A class to evaluate instance masks using PASCAL metrics."""

    def __init__(
        self,
        category_index,
        matching_iou_threshold=0.5,
        max_examples_to_draw=10,
        min_visualization_score_thresh=0.2
    ):
        super(PascalInstanceSegmentationEvaluator, self).__init__(
            category_index,
            matching_iou_threshold=matching_iou_threshold,
            evaluate_corlocs=False,
            metric_prefix='PascalMasks',
            use_weighted_mean_ap=False,
            evaluate_masks=True)


class WeightedPascalInstanceSegmentationEvaluator(ObjectDetectionEvaluator):
    """A class to evaluate instance masks using weighted PASCAL metrics.
    Weighted PASCAL metrics computes the mean average precision as the average
    precision given the scores and tp_fp_labels of all classes. In comparison,
    PASCAL metrics computes the mean average precision as the mean of the
    per-class average precisions.
    This definition is very similar to the mean of the per-class average
    precisions weighted by class frequency. However, they are typically not the
    same as the average precision is not a linear function of the scores and
    tp_fp_labels.
    """

    def __init__(
        self,
        category_index,
        matching_iou_threshold=0.5,
        max_examples_to_draw=10,
        min_visualization_score_thresh=0.2
    ):
        super(WeightedPascalInstanceSegmentationEvaluator, self).__init__(
            category_index,
            matching_iou_threshold=matching_iou_threshold,
            evaluate_corlocs=False,
            metric_prefix='WeightedPascalMasks',
            use_weighted_mean_ap=True,
            evaluate_masks=True)
