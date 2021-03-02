import numpy as np
import tensorflow as tf

from ..data import fields
from . import evaluator
from . import visualization


class SemSegEvaluator(evaluator.Evaluator):
    """Class to evaluate semantic segmentation metrics."""

    def __init__(
        self,
        category_index,
        ignore_label=255,
        include_metrics_per_category=False,
        max_examples_to_draw=10,
    ):
        """
        Args:
            category_index: a dict containing COCO-like category information keyed
                by the 'id' field of each category.
            ignore_label (int): value in semantic segmentation ground truth.
                Predictions for the corresponding pixels should be ignored.
        """
        super(SemSegEvaluator, self).__init__(category_index)
        self._ignore_label = ignore_label
        self._include_metrics_per_category = include_metrics_per_category
        self._num_classes = max(category_index.keys()) + 1
        self._N = self._num_classes + 1
        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)
        self._groundtruths = {}
        self._image_ids = {}
        self._summaries = []
        self._max_examples_to_draw = max_examples_to_draw

    def clear(self):
        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)
        self._groundtruths.clear()
        self._image_ids.clear()
        self._summaries = []

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """Adds groundtruth for a single image to be used for evaluation.
        Args:
            image_id: A unique string/integer identifier for the image.
            groundtruth_dict: A dictionary containing -
                fields.InputFields.sem_seg: unint8 numpy array of shape [height, width]
                    with values ranging between 0 ~ num_classes.
        Raises:
            ValueError: On adding groundtruth for an image more than once.
                Will also raise error if evaluate_masks and instance masks are not in
                groundtruth dictionary.
        """
        if image_id in self._image_ids:
            tf.logging.warning(
                'image %s has already been added to the ground truth database.',
                image_id)
            return
        sem_seg = groundtruth_dict[fields.InputFields.sem_seg]
        sem_seg[sem_seg == self._ignore_label] = self._num_classes

        if len(self._image_ids) < self._max_examples_to_draw:
            image = visualization.visualize_sem_seg_on_image_array(
                image=groundtruth_dict[fields.InputFields.image],
                sem_seg=sem_seg,
                category_index=self._category_index
            )
            self._summaries.append(
                tf.Summary.Value(
                    tag="{}/Groudtruth/Semantic".format(image_id),
                    image=tf.Summary.Image(
                        encoded_image_string=visualization.encode_image_array_as_png_str(image)))
            )
        self._groundtruths[image_id] = sem_seg
        self._image_ids[image_id] = False

    def add_single_predicted_image_info(self, image_id, result_dict):
        """Adds detections for a single image to be used for evaluation.
        Args:
        image_id: A unique string/integer identifier for the image.
        result_dict: A dictionary containing -
            fields.ResultFields.sem_seg: uint8 numpy array of shape [height, width]
                with values ranging between 0 ~ num_classes.
        Raises:
        ValueError: If detection masks are not in detections dictionary.
        """
        if image_id not in self._image_ids:
            raise ValueError('Missing groundtruth for image id: {}'.format(image_id))

        if self._image_ids[image_id]:
            tf.logging.warning(
                'image %s has already been added to the detection result database',
                image_id)
            return

        pred = result_dict[fields.ResultFields.sem_seg]

        if len(self._image_ids) <= self._max_examples_to_draw:
            image = visualization.visualize_sem_seg_on_image_array(
                image=result_dict[fields.InputFields.image],
                sem_seg=pred,
                category_index=self._category_index,
            )
            self._summaries.append(
                tf.Summary.Value(
                    tag="{}/Prediction/Semantic".format(image_id),
                    image=tf.Summary.Image(
                        encoded_image_string=visualization.encode_image_array_as_png_str(image)))
            )

        gt = self._groundtruths.pop(image_id)
        self._conf_matrix += np.bincount(
            self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N ** 2
        ).reshape(self._N, self._N)
        self._image_ids[image_id] = True

    def evaluate(self):
        """Compute evaluation result.
        Returns:
        A dictionary of metrics with the following fields -
            'SemSegPerformance/mIOU',
            'SemSegPerformance/fwIOU',
            'SemSegPerformance/mACC',
            'SemSegPerformance/pACC'
            if include_metrics_per_category, category specific results with keys
            of the form -
                'SemSegPerformanceByCategory/IOU/category',
                'SemSegPerformanceByCategory/ACC/category'.
        """
        acc = np.zeros(self._num_classes, dtype=np.float)
        iou = np.zeros(self._num_classes, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]

        metrics = {}
        metrics['SemSegPerformance/mIOU'] = np.sum(acc) / np.sum(acc_valid)
        metrics['SemSegPerformance/fwIOU'] = np.sum(iou) / np.sum(iou_valid)
        metrics['SemSegPerformance/mACC'] = np.sum(iou * class_weights)
        metrics['SemSegPerformance/pACC'] = np.sum(tp) / np.sum(pos_gt)

        if self._include_metrics_per_category:
            for category_id in self._category_index:
                category_name = self._category_index[category_id]["name"]
                metrics['SemSegPerformanceByCategory/IOU/{:s}'.format(
                    category_name)] = iou[category_id]
                metrics['SemSegPerformanceByCategory/ACC/{:s}'.format(
                    category_name)] = acc[category_id]

        for key, value in metrics.items():
            self._summaries.append(
                tf.Summary.Value(tag=key, simple_value=value)
            )
        return metrics, self._summaries
