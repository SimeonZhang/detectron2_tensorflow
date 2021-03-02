import os
import numpy as np
import tensorflow as tf
import json
import collections
from PIL import Image

from ..panopticapi.utils import id2rgb, rgb2id
from ..panopticapi.evaluation import pq_compute

from ..data import fields
from . import metrics
from . import per_image_evaluation
from . import visualization
from . import evaluator


def _get_bbox_from_single_mask(mask):
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    coco_bbox = [x1, y1, x2 - x1, y2 - y1]
    return np.asarray(coco_bbox, dtype=np.float32)


class CocoPanopticEvaluator(evaluator.Evaluator):

    def __init__(
        self,
        category_map,
        stuff_ignore_label,
        groundtruth_dir,
        prediction_dir,
        include_metrics_per_category=False,
        all_metrics_per_category=False,
        max_examples_to_draw=10,
    ):
        self._categories = list(category_map["category_index"].values())
        self._category_index = {cat['id']: cat for cat in self._categories}
        self._thing_contiguous_id_to_dataset_id = {
            v: int(k) for k, v in category_map["thing_id_map"].items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: int(k) for k, v in category_map["stuff_id_map"].items() if k != "0"
        }
        self._num_stuff_classes = category_map["num_stuff_classes"]
        self._stuff_ignore_label = stuff_ignore_label
        self._stuff_contiguous_id_to_dataset_id[self._stuff_ignore_label] = 0

        self._include_metrics_per_category = include_metrics_per_category,
        self._all_metrics_per_category = all_metrics_per_category
        self._annotation_id = 0

        gt_json = os.path.join(groundtruth_dir, "panoptic_val.json")
        with open(gt_json) as fid:
            groundtruth_data = json.load(fid)
        annotation_index = {}
        for annotation in groundtruth_data['annotations']:
            # image_id = os.path.splitext(annotation["file_name"])[0]
            image_id = annotation['image_id']
            annotation['image_id'] = image_id
            annotation_index[annotation['image_id']] = annotation
        self._annotation_index = annotation_index
        self._groundtruth_dir = os.path.join(groundtruth_dir, "panoptic_val")
        self._used_gt_json = os.path.join(prediction_dir, "panoptic_val.json")
        self._predictions_dir = os.path.join(prediction_dir, "predictions")
        self._predictions_json = os.path.join(prediction_dir, "predictions.json")
        if not os.path.exists(self._groundtruth_dir):
            raise ValueError(f"{self._groundtruth_dir} not exist!")
        if not os.path.exists(self._predictions_dir):
            os.makedirs(self._predictions_dir)

        self._max_examples_to_draw = max_examples_to_draw
        self._summaries = []

        self._image_ids_predicted = {}
        self._image_id_to_filename = {}
        self._predictions = []
        self._annotations = []

    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self._image_ids_predicted.clear()
        self._image_id_to_filename.clear()
        self._predictions = []
        self._annotations = []
        self._summaries = []

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict=None):
        """Adds groundtruth for a single image to be used for evaluation.
        Args:
            image_id: A unique string/integer identifier for the image.
            groundtruth_dict: A dictionary containing -
            InputFields.groundtruth_classes: integer numpy array of shape
            [num_boxes] containing 1-indexed groundtruth classes for the boxes.
            InputFields.groundtruth_instance_masks: uint8 numpy array of shape
            [num_boxes, image_height, image_width] containing groundtruth masks
            corresponding to the boxes. The elements of the array must be in
            {0, 1}.
        """
        if image_id in self._image_ids_predicted:
            tf.logging.warning(
                'image %s has already been added to the ground truth database.',
                image_id)
            return

        annotation = self._annotation_index[image_id]
        file_name_png = annotation["file_name"]
        self._image_id_to_filename[image_id] = file_name_png

        self._annotations.append(annotation)

        if len(self._image_ids_predicted) < self._max_examples_to_draw:
            file_name_path = os.path.join(self._groundtruth_dir, file_name_png)
            panoptic_seg = rgb2id(np.uint8(Image.open(file_name_path)))
            image = visualization.visualize_panoptic_seg_on_image_array(
                image=groundtruth_dict[fields.InputFields.image],
                panoptic_seg=panoptic_seg,
                segments_info=annotation["segments_info"],
                category_index=self._category_index
            )
            self._summaries.append(
                tf.Summary.Value(
                    tag="{}/Groudtruth/Panoptic".format(image_id),
                    image=tf.Summary.Image(
                        encoded_image_string=visualization.encode_image_array_as_png_str(image)))
            )

        self._image_ids_predicted[image_id] = False

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
        if image_id not in self._image_ids_predicted:
            tf.logging.warning('Ignoring detection with image id %s since there was '
                               'no groundtruth added', image_id)
            return

        if self._image_ids_predicted[image_id]:
            tf.logging.warning('Ignoring detection with image id %s since it was '
                               'previously added', image_id)
            return

        assert "panoptic_seg" in result_dict
        panoptic_result = result_dict["panoptic_seg"]
        panoptic_seg = panoptic_result["panoptic_seg"]

        segments_info = []

        num_segments = panoptic_result["id"].shape[0]

        for idx in range(num_segments):
            segment_id = int(panoptic_result["id"][idx])
            category_id = int(panoptic_result["category_id"][idx])
            if panoptic_result["isthing"][idx]:
                segments_info.append(
                    {
                        "id": segment_id,
                        "category_id": self._thing_contiguous_id_to_dataset_id[category_id],
                    }
                )
            else:
                segments_info.append(
                    {
                        "id": segment_id,
                        "category_id": self._stuff_contiguous_id_to_dataset_id[category_id],
                    }
                )

        file_name_png = self._image_id_to_filename[image_id]
        self._predictions.append(
            {
                "image_id": image_id,
                "file_name": file_name_png,
                "segments_info": segments_info,
            }
        )

        if len(self._image_ids_predicted) <= self._max_examples_to_draw:
            image = visualization.visualize_panoptic_seg_on_image_array(
                image=result_dict[fields.InputFields.image],
                panoptic_seg=panoptic_seg,
                segments_info=segments_info,
                category_index=self._category_index,
            )
            self._summaries.append(
                tf.Summary.Value(
                    tag="{}/Prediction/Panoptic".format(image_id),
                    image=tf.Summary.Image(
                        encoded_image_string=visualization.encode_image_array_as_png_str(image)))
            )

        with open(os.path.join(self._predictions_dir, file_name_png), "w+b") as out:
            Image.fromarray(id2rgb(panoptic_seg), mode="RGB").save(out, format="PNG")
        self._image_ids_predicted[image_id] = True

    def evaluate(self):
        with open(self._used_gt_json, "w") as gt_json_file:
            gt_json_obj = {
                "annotations": self._annotations,
                "categories": self._categories,
            }
            json.dump(gt_json_obj, gt_json_file, indent=2)

        with open(self._predictions_json, "w") as pred_json_file:
            pred_json_obj = {
                "annotations": self._predictions,
            }
            json.dump(pred_json_obj, pred_json_file, indent=2)

        pq_res = pq_compute(
            self._used_gt_json,
            self._predictions_json,
            gt_folder=self._groundtruth_dir,
            pred_folder=self._predictions_dir,
        )

        metrics = collections.OrderedDict([
            ('Panoptic/PQ', pq_res["All"]["pq"]),
            ('Panoptic/SQ', pq_res["All"]["sq"]),
            ('Panoptic/RQ', pq_res["All"]["rq"]),
            ('Panoptic/N', pq_res["All"]["n"]),
            ('Panoptic/PQ_th', pq_res["Things"]["pq"]),
            ('Panoptic/SQ_th', pq_res["Things"]["sq"]),
            ('Panoptic/RQ_th', pq_res["Things"]["rq"]),
            ('Panoptic/N_th', pq_res["Things"]["n"]),
            ('Panoptic/PQ_st', pq_res["Stuff"]["pq"]),
            ('Panoptic/SQ_st', pq_res["Stuff"]["sq"]),
            ('Panoptic/RQ_st', pq_res["Stuff"]["rq"]),
            ('Panoptic/N_st', pq_res["Stuff"]["n"]),
        ])

        if self._include_metrics_per_category:
            for category in self._categories:
                category_id = category["id"]
                category_name = category["name"]
                metrics["PerformanceByCategory/PQ/{}".format(
                    category_name)] = pq_res["per_class"][category_id]["pq"]
                if self._all_metrics_per_category:
                    metrics['PanopticByCategory/PQ/{}'.format(
                        category_name)] = pq_res["per_class"][category_id]["pq"]
                    metrics['PanopticByCategory/SQ/{}'.format(
                        category_name)] = pq_res["per_class"][category_id]["sq"]
                    metrics['PanopticByCategory/RQ/{}'.format(
                        category_name)] = pq_res["per_class"][category_id]["rq"]
                    metrics['PanopticByCategory/N/{}'.format(
                        category_name)] = pq_res["per_class"][category_id]["n"]

        for key, value in metrics.items():
            self._summaries.append(
                tf.Summary.Value(tag=key, simple_value=value)
            )
        return metrics, self._summaries
