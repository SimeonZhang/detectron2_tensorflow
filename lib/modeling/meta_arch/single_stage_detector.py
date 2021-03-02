import tensorflow as tf

from ...layers import Layer
from ..backbone import build_backbone
from ..necks import build_neck
from ..single_stage_heads import build_single_stage_head
from ...data import fields
from ...structures import ImageList
from .build import META_ARCH_REGISTRY


__all__ = ["SingleStageDetector"]


@META_ARCH_REGISTRY.register()
class SingleStageDetector(Layer):
    """
    Generalized single stage detector.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)

        self.backbone = build_backbone(cfg, scope="backbone")
        self.neck = build_neck(cfg, self.backbone.output_shape(), scope="neck")
        self.detector = build_single_stage_head(cfg, self.neck.output_shape(), scope="head")

        pixel_mean = tf.convert_to_tensor(cfg.MODEL.PIXEL_MEAN, tf.float32)
        pixel_std = tf.convert_to_tensor(cfg.MODEL.PIXEL_STD, tf.float32)
        self.input_format = cfg.MODEL.INPUT_FORMAT
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def call(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs:
            gt_instances = batched_inputs["instances"]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = self.neck(features)
        results, losses = self.detector(images, features, gt_instances)

        if self.training:
            return losses
        else:
            result_fields = fields.ResultFields
            detected_results = {
                result_fields.boxes: results.boxes,
                result_fields.classes: results.get_field("pred_classes"),
                result_fields.scores: results.get_field("scores"),
                result_fields.is_valid: results.get_field("is_valid"),
            }
            if results.has_field("pred_masks"):
                detected_results[result_fields.masks] = results.get_field("pred_masks")
            return {"instances": detected_results}

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
