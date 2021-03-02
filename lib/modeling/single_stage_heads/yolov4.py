import tensorflow as tf
import math

from ...layers import (
    Layer,
    Sequential,
    Conv2D,
    get_norm,
    get_activation
)
from ..anchor_generator import build_anchor_generator
from ...utils.arg_scope import arg_scope
from .build import SINGLE_STAGE_HEADS_REGISTRY
from .yolov4_outputs import YOLOV4Outputs
from ..matcher import YOLOMatcher

slim = tf.contrib.slim

__all__ = ["YOLOV4Head"]


@SINGLE_STAGE_HEADS_REGISTRY.register()
class YOLOV4Head(Layer):
    """
    Implement YOLO V4 (https://arxiv.org/abs/2004.10934).
    """

    def __init__(self, cfg, input_shape, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = cfg.MODEL.SINGLE_STAGE_HEAD.NUM_CLASSES
        self.in_features = cfg.MODEL.SINGLE_STAGE_HEAD.IN_FEATURES
        self.in_strides = [input_shape[f].stride for f in self.in_features]
        self.in_channels = [input_shape[f].channels for f in self.in_features]

        feature_shapes = [input_shape[f] for f in self.in_features]
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)
        self.head = YOLOV4Tower(
            cfg, self.in_channels, self.anchor_generator.num_cell_anchors, scope="head"
        )

        # Matching and loss
        self.matcher = YOLOMatcher(cfg.MODEL.SINGLE_STAGE_HEAD.IOU_THRESHOLDS[0])
        self.cls_normalizer = cfg.MODEL.YOLOV4.CLS_NORMALIZER
        self.iou_normalizer = cfg.MODEL.YOLOV4.IOU_NORMALIZER

        # Inference parameters:
        self.score_threshold = cfg.MODEL.YOLOV4.SCORE_THRESH_TEST
        self.nms_threshold = cfg.MODEL.YOLOV4.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        self.scale_yx = cfg.MODEL.YOLOV4.SCALE_YX

    def call(self, images, features, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (BoxList): Dense `BoxList` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (BoxList, optional): lDense `BoxList` contains the ground-truth per-instance 
                annotations for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: the ground-truth mask of the instance.
        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        features = [features[f] for f in self.in_features]
        pred_logits = self.head(features)
        # print_ops = []
        # for f, logits in zip(self.in_features, features):
        #     print_ops.append(tf.print(f, "\n", tf.shape(logits), "\n", logits))
        # with tf.control_dependencies(print_ops):
        #     pred_logits[-1] = tf.identity(pred_logits[-1])

        outputs = YOLOV4Outputs(
            self.matcher,
            self.anchor_generator,
            pred_logits,
            self.num_classes,
            self.cls_normalizer,
            self.iou_normalizer,
            self.scale_yx,
            self.score_threshold,
            self.nms_threshold,
            self.max_detections_per_image,
            images,
            targets
        )

        if self.training:
            losses = outputs.losses()
            return None, losses
        else:
            results = outputs.inference()
            return results, {}


class YOLOV4Tower(Layer):
    """
    The head used in YOLO V4 for object classification and box regression.
    """

    def __init__(self, cfg, in_channels, num_anchors, **kwargs):
        super().__init__(**kwargs)
        in_channels = in_channels
        num_classes = cfg.MODEL.SINGLE_STAGE_HEAD.NUM_CLASSES
        conv_dims = cfg.MODEL.YOLOV4.CONV_DIMS
        normalizer = get_norm(cfg.MODEL.YOLOV4.NORM)
        activation = get_activation(cfg.MODEL.YOLOV4.ACTIVATION, alpha=0.1)
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        self.heads = [[], [], []]
        num_mids = [0, 2, 2]
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            with arg_scope(
                [Conv2D],
                stride=1,
                normalizer=normalizer,
                normalizer_params={"scope": "norm"},
                use_bias=not normalizer,
                activation=activation,
                padding='SAME',
                weights_initializer=tf.random_normal_initializer(stddev=0.01),
            ):
                for i in range(3):
                    head_dims = 2 ** i * conv_dims
                    self.heads[i].append(
                        Conv2D(
                            in_channels=in_channels[i],
                            out_channels=head_dims,
                            kernel_size=3,
                            scope="conv{:d}".format(i + 1)
                        )
                    )
                    self.heads[i].append(
                        Conv2D(
                            in_channels=head_dims,
                            out_channels=num_anchors * (num_classes + 5),
                            kernel_size=1,
                            normalizer=None,
                            use_bias=True,
                            activation=None,
                            scope="pred{:d}".format(i + 1)
                        )
                    )
                    self.heads[i] = Sequential(self.heads[i])

    def call(self, features):
        """
        Arguments:
            features [Tensor]: feature map tensors.
        Returns:
        """
        res = []
        for head, feature in zip(self.heads, features):
            res.append(head(feature))
        return res
