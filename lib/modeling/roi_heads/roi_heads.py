import tensorflow as tf
from typing import Dict

from ...utils.registry import Registry
from ...structures import box_list
from ...structures import box_list_ops
from ...structures import mask_ops
from ...layers import Layer, ShapeSpec
from ..sampling import subsample_labels
from ..poolers import ROIPooler
from ..matcher import Matcher
from ..box_regression import Box2BoxTransform
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..backbone.resnet import resnet_arg_scope, Stage
from ..backbone.blocks import BottleneckBlock
from . import fast_rcnn
from . import mask_head
from . import box_head

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
"""
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and perform per-region computation.
"""


def build_roi_heads(cfg, input_shape, **kwargs):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape, **kwargs)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a , each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.
    Args:
        proposals (SparseBoxList):
        bg_label: label index of background class.
    Returns:
        SparseBoxList
    """
    assert isinstance(proposals, box_list.SparseBoxList)
    assert proposals.data.has_field("gt_classes")
    gt_classes = proposals.data.get_field("gt_classes")
    fg_inds = tf.where(
        tf.logical_and(
            tf.not_equal(gt_classes, -1), tf.not_equal(gt_classes, bg_label)
        )
    )[:, 0]
    fg_proposals = box_list.SparseBoxList(
        tf.gather(proposals.indices, fg_inds),
        box_list_ops.gather(proposals.data, fg_inds),
        proposals.dense_shape,
    )
    fg_proposals.set_tracking(
        'image_shape', proposals.get_tracking('image_shape')
    )
    return fg_proposals, fg_inds


class ROIHeads(Layer):
    """
    ROIHeads perform all per-region computation in an R-CNN.
    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.
    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec], **kwargs):
        super(ROIHeads, self).__init__(**kwargs)

        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img = cfg.TEST.DETECTIONS_PER_IMAGE
        self.test_nms_cls_agnostic = cfg.MODEL.ROI_HEADS.NMS_CLS_AGNOSTIC
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assign
        training labels to the lproposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.
        Args:
            See :meth:`ROIHeads.forward`
        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        def label_and_sample_single_image(inputs):
            proposals_dict, targets_dict = inputs

            is_valid = targets_dict.pop('is_valid')
            is_crowd = tf.cast(targets_dict.pop('gt_is_crowd'), tf.bool)
            difficult = tf.cast(targets_dict.pop('gt_difficult'), tf.bool)
            valid_bool = is_valid & ~is_crowd & ~difficult
            target_boxlist = box_list.BoxList.from_tensor_dict(targets_dict)
            valid_target_boxlist = box_list_ops.boolean_mask(target_boxlist, valid_bool)
            crowd_target_boxlist = box_list_ops.boolean_mask(target_boxlist, is_crowd)
            difficult_target_boxlist = box_list_ops.boolean_mask(target_boxlist, difficult)

            is_valid_proposals = proposals_dict.pop('is_valid')
            image_shape = proposals_dict.pop('image_shape')
            proposal_boxlist = box_list.BoxList.from_tensor_dict(proposals_dict)
            proposal_boxlist = box_list_ops.boolean_mask(proposal_boxlist, is_valid_proposals)

            match_quality_matrix = box_list_ops.pairwise_iou(
                valid_target_boxlist, proposal_boxlist
            )
            crowd_matrix = box_list_ops.pairwise_iou(
                crowd_target_boxlist, proposal_boxlist
            )
            difficult_matrix = box_list_ops.pairwise_iou(
                difficult_target_boxlist, proposal_boxlist
            )
            matched_idxs, proposals_labels = self.proposal_matcher(
                match_quality_matrix, crowd_matrix, difficult_matrix)

            fg_inds = tf.cast(
                tf.where(tf.equal(proposals_labels, 1))[:, 0], tf.int32
            )
            bg_inds = tf.cast(
                tf.where(tf.equal(proposals_labels, 0))[:, 0], tf.int32
            )
            ig_inds = tf.cast(
                tf.where(tf.equal(proposals_labels, -1))[:, 0], tf.int32
            )

            fg_targets = tf.gather(matched_idxs, fg_inds)
            gt_classes = tf.dynamic_stitch(
                [fg_inds, bg_inds, ig_inds],
                [
                    tf.gather(valid_target_boxlist.get_field('gt_classes'), fg_targets),
                    tf.zeros_like(bg_inds, dtype=tf.int64) + self.num_classes,
                    tf.zeros_like(ig_inds, dtype=tf.int64) - 1,
                ]
            )
            sampled_fg_inds, sampled_bg_inds = subsample_labels(
                gt_classes,
                self.batch_size_per_image,
                self.positive_sample_fraction,
                self.num_classes,
            )
            sampled_inds = tf.concat([sampled_fg_inds, sampled_bg_inds], axis=0)
            sampled_proposal_boxlist = box_list_ops.gather(
                proposal_boxlist, sampled_inds
            )
            sampled_proposal_boxlist.add_field(
                'gt_classes', tf.gather(gt_classes, sampled_inds)
            )
            sampled_proposal_boxlist.add_field(
                'is_valid', tf.ones_like(sampled_inds, tf.bool)
            )

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            sampled_targets = tf.gather(matched_idxs, sampled_inds)
            for field in valid_target_boxlist.get_extra_fields():
                if field.startswith('gt_') and not sampled_proposal_boxlist.has_field(field):
                    sampled_proposal_boxlist.add_field(
                        field, tf.gather(valid_target_boxlist.get_field(field), sampled_targets)
                    )
            sampled_proposal_boxlist.add_field(
                'gt_boxes', tf.gather(valid_target_boxlist.boxes, sampled_targets)
            )

            sampled_proposal_boxlist = box_list_ops.pad_or_clip_boxlist(
                sampled_proposal_boxlist, self.batch_size_per_image
            )
            sampled_proposal_dict = sampled_proposal_boxlist.as_tensor_dict()
            sampled_proposal_dict['image_shape'] = image_shape
            return sampled_proposal_dict

        proposals_dict = proposals.as_tensor_dict()
        targets_dict = targets.as_tensor_dict()
        dtype = {
            field: value.dtype for field, value in proposals_dict.items()
        }
        for field, value in targets_dict.items():
            if field.startswith("gt_") and field not in dtype and(
                field not in ["gt_is_crowd", "gt_difficult"]
            ):
                dtype[field] = value.dtype
        dtype['gt_boxes'] = targets.boxes.dtype
        sampled_proposal_dict = tf.map_fn(
            label_and_sample_single_image,
            [proposals_dict, targets_dict],
            dtype=dtype,
        )
        return box_list.BoxList.from_tensor_dict(sampled_proposal_dict, trackings=['image_shape'])

    def call(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are height, width, and channels, which may
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
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape, **kwargs):
        super().__init__(cfg, input_shape, **kwargs)

        assert len(self.in_features) == 1

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on = cfg.MODEL.MASK_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            self.res5, out_channels = self._build_res5_block(cfg)
            self.box_predictor = fast_rcnn.FastRCNNOutputLayers(
                out_channels, self.num_classes, self.cls_agnostic_bbox_reg, scope="fastrcnn"
            )

            if self.mask_on:
                self.mask_head = mask_head.build_mask_head(
                    cfg,
                    ShapeSpec(
                        channels=out_channels, width=pooler_resolution, height=pooler_resolution),
                    scope="mask_head"
                )

    def _build_res5_block(self, cfg):

        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group * stage_channel_factor
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."

        block_kwargs = {
            "in_channels": out_channels // 2,
            "out_channels": out_channels,
            "bottleneck_channels": bottleneck_channels,
            "num_groups": num_groups,
            "stride_in_1x1": stride_in_1x1
        }
        with resnet_arg_scope(False, norm):
            stage = Stage(
                BottleneckBlock,
                block_kwargs=block_kwargs,
                num_blocks=3,
                first_stride=2,
                scope='res5'
            )
        return stage, out_channels

    def _shared_roi_transform(self, features, proposals):
        x = self.pooler(features, proposals)
        return self.res5(x)

    def call(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.call`.
        """
        image_shape = tf.shape(images.tensor)[1:3]

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposals = box_list.SparseBoxList.from_dense(proposals)
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposals
        )
        feature_pooled = tf.reduce_mean(box_features, axis=[1, 2])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled)
        del feature_pooled

        outputs = fast_rcnn.FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()
            if self.mask_on:
                proposals, fg_inds = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = tf.gather(box_features, fg_inds)
                del box_features
                mask_logits = self.mask_head(mask_features)
                proposals.data.add_field("deconv_features", deconv_features)
                losses["loss_mask"] = mask_rcnn_loss(mask_logits, proposals)
            return proposals, losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh,
                self.test_detections_per_img, self.test_nms_cls_agnostic
            )
            pred_instances = self.forward_with_given_boxes(features, pred_instances, image_shape)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances, image_shape):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.
        Args:
            features: same as in `forward()`
            instances (SparseBoxList): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.
        Returns:
            instances (BoxList):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances.has_field("pred_classes")

        if self.mask_on:
            instances = box_list.SparseBoxList.from_dense(instances)
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, instances)
            deconv_features, mask_logits = self.mask_head(x)
            mask_head.mask_rcnn_inference(mask_logits, instances)
            instances.data.add_field("deconv_features", deconv_features)
            instances = instances.to_dense()
        return instances


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.
    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`call()` or a head.
    """

    def __init__(self, cfg, input_shape, **kwargs):
        super(StandardROIHeads, self).__init__(cfg, input_shape, **kwargs)
        self._init_box_head(cfg)
        self._init_mask_head(cfg)

    def _init_box_head(self, cfg):
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            self.box_head = box_head.build_box_head(
                cfg,
                ShapeSpec(
                    channels=in_channels, height=pooler_resolution, width=pooler_resolution
                ),
                scope="box_head"
            )
            self.box_predictor = fast_rcnn.FastRCNNOutputLayers(
                self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg,
                scope="box_predictor"
            )

    def _init_mask_head(self, cfg):
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        self.use_mini_masks = cfg.TRANSFORM.RESIZE.USE_MINI_MASKS
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            self.mask_head = mask_head.build_mask_head(
                cfg,
                ShapeSpec(
                    channels=in_channels, width=pooler_resolution, height=pooler_resolution
                ),
                scope="mask_head"
            )

    def call(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.call`.
        """
        image_shape = tf.shape(images.tensor)[1:3]

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]
        proposals = box_list.SparseBoxList.from_dense(proposals)

        if self.training:
            losses = self._forward_box(features_list, proposals)
            # During training the proposals used by the box head are
            # used by the mask heads.
            loss_mask, deconv_features = self._forward_mask(features_list, proposals)
            if deconv_features is not None:
                proposals.data.add_field("deconv_features", deconv_features)
            losses.update(loss_mask)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances, image_shape)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances, image_shape):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.
        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.
        Args:
            features: same as in `call()`
            instances (SparseBoxList): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.
        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances.has_field("pred_classes")
        features = [features[f] for f in self.in_features]

        if self.mask_on:
            instances = box_list.SparseBoxList.from_dense(instances)
            instances, deconv_features = self._forward_mask(features, instances)
            instances.data.add_field("deconv_features", deconv_features)
            instances = instances.to_dense()
        return instances

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.
        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (SparseBoxList): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, proposals)
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = fast_rcnn.FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh,
                self.test_detections_per_img, self.test_nms_cls_agnostic
            )
            return pred_instances

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return ({}, None) if self.training else (instances, None)

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            mask_features = self.mask_pooler(features, proposals)
            deconv_features, mask_logits = self.mask_head(mask_features)
            loss_mask = mask_head.mask_rcnn_loss(mask_logits, proposals, self.use_mini_masks)
            return {"loss_mask": loss_mask}, deconv_features
        else:
            mask_features = self.mask_pooler(features, instances)
            deconv_features, mask_logits = self.mask_head(mask_features)
            mask_head.mask_rcnn_inference(mask_logits, instances)
            return instances, deconv_features
