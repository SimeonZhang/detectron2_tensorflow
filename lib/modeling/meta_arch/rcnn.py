import tensorflow as tf

from ...data import fields
from ...layers import Layer
from ...structures import ImageList, box_list
from ..backbone import build_backbone
from ..necks import build_neck
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from ..postprocessing import detector_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(Layer):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg, scope="backbone")
        self.neck = build_neck(cfg, self.backbone.output_shape(), scope="neck")
        self.proposal_generator = build_proposal_generator(
            cfg, self.neck.output_shape(), scope="proposal_generator")
        self.roi_heads = build_roi_heads(cfg, self.neck.output_shape(), scope="roi_heads")

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        pixel_mean = tf.convert_to_tensor(cfg.MODEL.PIXEL_MEAN, tf.float32)
        pixel_std = tf.convert_to_tensor(cfg.MODEL.PIXEL_STD, tf.float32)
        self.input_format = cfg.MODEL.INPUT_FORMAT
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.segmentation_output_format = cfg.MODEL.SEGMENTATION_OUTPUT.FORMAT
        self.segmentation_output_resolution = cfg.MODEL.SEGMENTATION_OUTPUT.FIXED_RESOLUTION

    def call(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict:
                Dict is the outputs for input images.
                The dict contains one key "instances" whose value is a :class:`Dict`.
                The :class:`Dict` object has the following keys:
                    "boxes", "classes", "scores", "masks", 
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs:
            gt_instances = batched_inputs["instances"]
        elif "targets" in batched_inputs:
            tf.logging.warn(
                "'targets' in the model inputs is now renamed to 'instances'!"
            )
            gt_instances = batched_inputs["targets"]
        else:
            gt_instances = None

        features = self.neck(self.backbone(images.tensor))

        if self.proposal_generator:
            proposals, proposal_losses, _ = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs
            proposals = batched_inputs["proposals"]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or BoxList): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
        Returns:
            same as in :meth:`call`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.neck(self.backbone(images.tensor))

        if detected_instances is None:
            if self.proposal_generator:
                proposals, *_ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs
                proposals = batched_inputs["proposals"]
            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = box_list.SparseBoxList.from_dense(detected_instances)
            results, _ = self.roi_heads.forward_with_given_boxes(
                features, detected_instances, tf.shape(images.tensor)[1:3]
            )

        if self.segmentation_output_format != "raw":
            if self.segmentation_output_format == "fixed":
                output_shape = [
                    self.segmentation_output_resolution, self.segmentation_output_resolution
                ]
            elif self.segmentation_output_format == "conventional":
                output_shape = tf.shape(images.tensor)[1:3]
            results = detector_postprocess(
                results, output_shape, self.segmentation_output_format, images.image_shapes
            )

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


@META_ARCH_REGISTRY.register()
class ProposalNetwork(Layer):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg, self.backbone.output_shape(), scope="neck")
        self.proposal_generator = build_proposal_generator(
            cfg, self.neck.output_shape(), scope="proposal_generator")

        pixel_mean = tf.convert_to_tensor(cfg.MODEL.PIXEL_MEAN, tf.float32)
        pixel_std = tf.convert_to_tensor(cfg.MODEL.PIXEL_STD, tf.float32)
        self.input_format = cfg.MODEL.INPUT_FORMAT
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def call(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`
        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = self.preprocess_image(batched_inputs)
        features = self.neck(self.backbone(images.tensor))

        if "instances" in batched_inputs:
            gt_instances = batched_inputs["instances"]
        elif "targets" in batched_inputs:
            tf.logging.warn(
                "'targets' in the model inputs is now renamed to 'instances'!"
            )
            gt_instances = batched_inputs["targets"]
        else:
            gt_instances = None

        proposals, proposal_losses, _ = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        result_fields = fields.ResultFields
        results = {
            result_fields.boxes: proposals.boxes,
            result_fields.is_valid: proposals.get_field("is_valid"),
        }
        scores = proposals.get_field("objectness_logits")
        classes = tf.zeros_like(scores, dtype=tf.int64)
        results[result_fields.classes] = classes
        results[result_fields.scores] = tf.nn.sigmoid(scores)
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
