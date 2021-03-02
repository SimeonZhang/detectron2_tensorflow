import os
import re
import tensorflow as tf
import json
import copy
from tensorflow.python.tools import freeze_graph

from ..data import fields
from .placeholder import build_single_input_placeholder
from ..modeling.meta_arch import build_model
from . import postprocessing
from .base import Exporter, EXPORTER_REGISTRY


@EXPORTER_REGISTRY.register()
class Detection(Exporter):
    """Exporter exporting a model with the interface of single batch detection."""

    def build_inference_graph(self, cfg):
        result_fields = fields.ResultFields
        serving_fields = fields.ServingFields
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        id_map = self.category_map["thing_id_map"]
        class_names = ["thing"] * num_classes
        for idx, item in self.category_map["category_index"].items():
            if item["isthing"]:
                class_names[id_map[idx]] = item["name"]

        encoded_input_image, decoded_input_image, batched_inputs = (
            build_single_input_placeholder(
                cfg,
                encoded_image_name=self.tensor_prefix + "encoded_image_string_tensor",
                decoded_image_name=self.tensor_prefix + "image_tensor",
                expand_batch_dimension=True
            )
        )
        input_shapes = batched_inputs["image_shape"]
        model_fn = build_model(cfg)
        results = model_fn(batched_inputs)

        results = postprocessing.detector_postprocess(
            results["instances"], input_shapes, class_names, self.label_offset
        )
        inputs = {"inputs": decoded_input_image}

        output_node_map = {
            result_fields.boxes: "detection_boxes",
            result_fields.classes: "detection_classes",
            result_fields.scores: "detection_scores",
            result_fields.class_names: "detection_class_names",
            "num_detections": "num_detections"
        }
        outputs = {}
        for name in results:
            if name in output_node_map:
                new_name = self.tensor_prefix + output_node_map[name]
                outputs[new_name] = tf.identity(results[name], new_name)

        return inputs, outputs

    def build_signature_def_map(self, inputs, outputs):
        tensor_info_inputs = {
            "inputs": tf.saved_model.utils.build_tensor_info(inputs["inputs"])
        }
        signature_def_map = {}

        detection_outputs_tensor_info = {
            'boxes': tf.saved_model.utils.build_tensor_info(outputs[self.tensor_prefix + 'detection_boxes']),
            'scores': tf.saved_model.utils.build_tensor_info(outputs[self.tensor_prefix + 'detection_scores']),
            'classes': tf.saved_model.utils.build_tensor_info(outputs[self.tensor_prefix + 'detection_classes']),
            'class_names': tf.saved_model.utils.build_tensor_info(outputs[self.tensor_prefix + 'detection_class_names']),
            'num_detections': tf.saved_model.utils.build_tensor_info(outputs[self.tensor_prefix + 'num_detections']),
        }

        detection_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=tensor_info_inputs,
                outputs=detection_outputs_tensor_info,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        signature_def_map["detection_classify"] = detection_signature

        return signature_def_map
