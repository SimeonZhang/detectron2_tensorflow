from typing import Dict, List
import tensorflow as tf

from ...layers import ShapeSpec, Layer, Conv2D
from ...utils.registry import Registry

from ..anchor_generator import build_anchor_generator
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from .build import PROPOSAL_GENERATOR_REGISTRY
from .rpn_outputs import RPNOutputs, find_top_rpn_proposals
from ...structures import box_list, box_list_ops
from ...utils.arg_scope import arg_scope

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")
"""
Registry for RPN heads, which take feature maps and perform
objectness classification and bounding box regression for anchors.
"""


def build_rpn_head(cfg, input_shape, **kwargs):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape, **kwargs)


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(Layer):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec], **kwargs):
        super().__init__(**kwargs)

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_cell_anchors)) == 1
        ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            # 3x3 conv for the hidden representation
            self.conv = Conv2D(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                activation=tf.nn.relu,
                scope='share'
            )
            # 1x1 conv for predicting objectness logits
            self.objectness_logits = Conv2D(
                in_channels,
                num_cell_anchors,
                kernel_size=1,
                stride=1,
                scope='objectness_logits'
            )
            # 1x1 conv for predicting box2box transform deltas
            self.anchor_deltas = Conv2D(
                in_channels,
                num_cell_anchors * box_dim,
                kernel_size=1,
                stride=1,
                scope='anchor_deltas'
            )

    def call(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        rpn_features = []
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            share = self.conv(x)
            rpn_features.append(share)
            pred_objectness_logits.append(self.objectness_logits(share))
            pred_anchor_deltas.append(self.anchor_deltas(share))
        return rpn_features, pred_objectness_logits, pred_anchor_deltas


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN(Layer):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec], **kwargs):
        super().__init__(**kwargs)

        # fmt: off
        self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh = cfg.MODEL.RPN.NMS_THRESH
        self.batch_size_per_image = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta = cfg.MODEL.RPN.SMOOTH_L1_BETA
        self.loss_weight = cfg.MODEL.RPN.LOSS_WEIGHT
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH

        self.anchor_generator = build_anchor_generator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        with tf.variable_scope(self.scope, auxiliary_name_scope=False), \
            arg_scope(
                [Conv2D],
                weights_initializer=tf.random_normal_initializer(stddev=0.01)):
            self.rpn_head = build_rpn_head(
                cfg, [input_shape[f] for f in self.in_features], scope="rpn_head")

    def call(self, images, features, gt_instances=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are height, width, and channels, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (BoxList, optional)
        Returns:
            proposals: BoxList or None
            loss: dict[Tensor]
        """
        features = [features[f] for f in self.in_features]
        rpn_features, pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        anchors = self.anchor_generator(features)
        # TODO: The anchors only depend on the feature map shape; there's probably
        # an opportunity for some optimizations (e.g., caching anchors).
        outputs = RPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_instances,
            self.smooth_l1_beta,
        )

        if self.training:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
        else:
            losses = {}

        # Find the top proposals by applying NMS and removing boxes that
        # are too small. The proposals are treated as fixed for approximate
        # joint training with roi heads. This approach ignores the derivative
        # w.r.t. the proposal boxes’ coordinates that are also network
        # responses, so is approximate.
        proposals = find_top_rpn_proposals(
            outputs.predict_proposals(),
            outputs.predict_objectness_logits(),
            images,
            self.nms_thresh,
            self.pre_nms_topk[self.training],
            self.post_nms_topk[self.training],
            self.min_box_side_len,
        )
        rpn_features = {f: rpn_features[i] for i, f in enumerate(self.in_features)}
        return proposals, losses, rpn_features
