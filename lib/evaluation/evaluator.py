"""
Evaluator is a class which manages ground truth information of a input dataset,
and computes frequently used detection/semantic segmentation/panoptic segmentation
metrics such as Precision, Recall, CorLoc of the provided detection results.
It supports the following operations:
1) Add ground truth information of images sequentially.
2) Add result of images sequentially.
3) Evaluate metrics on already inserted results.
4) Write evaluation result into a pickle file for future processing or
   visualization.
Note: This module operates on numpy boxes and box lists.
"""

from abc import ABCMeta, abstractmethod


class Evaluator(object):
    """Interface for Evaluator classes.
    Example usage of the Evaluator:
    ------------------------------
    Evaluator = Evaluator(category_index)
    # Detections and groundtruth for image 1.
    Evaluator.add_single_groundtruth_image_info(...)
    Evaluator.add_single_predicted_image_info(...)
    # Detections and groundtruth for image 2.
    Evaluator.add_single_groundtruth_image_info(...)
    Evaluator.add_single_predicted_image_info(...)
    metrics_dict = Evaluator.evaluate()
    """
    __metaclass__ = ABCMeta

    def __init__(self, category_index):
        """Constructor.
        Args:
          category_index: a dict containing COCO-like category information keyed
            by the 'id' field of each category.
        """
        self._category_index = category_index

    @abstractmethod
    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """Adds groundtruth for a single image to be used for evaluation.
        Args:
            image_id: A unique string/integer identifier for the image.
            groundtruth_dict: A dictionary of groundtruth numpy arrays required
            for evaluations.
        """
        pass

    @abstractmethod
    def add_single_predicted_image_info(self, image_id, result_dict):
        """Adds results for a single image to be used for evaluation.
        Args:
            image_id: A unique string/integer identifier for the image.
            result_dict: A dictionary of result numpy arrays required
            for evaluation.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluates results and returns a dictionary of metrics."""
        pass

    @abstractmethod
    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        pass
