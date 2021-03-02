import tensorflow as tf

from ...structures import box_list
from ...structures import box_list_ops
from ...layers import ShapeSpec
from ..box_regression import Box2BoxTransform
from ..poolers import ROIPooler
from ..matcher import Matcher
from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference
from .box_head import build_box_head
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads


@ROI_HEADS_REGISTRY.register()
class CascadeROIHeads(StandardROIHeads):

    def _init_box_head(self, cfg):
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        self.num_cascade_stages = len(cascade_ious)
        assert len(cascade_bbox_reg_weights) == self.num_cascade_stages
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CascadeROIHeads only support class-agnositc regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]

        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        if self.training:
            @tf.custom_gradient
            def scale_gradient(x):
                return x, lambda dy: dy * (1.0 / self.num_cascade_stages)
            self.scale_gradient = scale_gradient
        else:
            self.scale_gradient = tf.identity

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )

        self.box_head = []
        self.box_predictor = []
        self.box2box_transform = []
        self.proposal_matchers = []
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            for k in range(self.num_cascade_stages):
                box_head = build_box_head(cfg, pooled_shape, scope='box_head_stage{}'.format(k + 1))
                self.box_head.append(box_head)
                self.box_predictor.append(
                    FastRCNNOutputLayers(
                        box_head.output_size,
                        self.num_classes,
                        cls_agnostic_bbox_reg=True,
                        scope='box_predictor_stage{}'.format(k + 1)
                    )
                )
                self.box2box_transform.append(Box2BoxTransform(weights=cascade_bbox_reg_weights[k]))

                if k == 0:
                    # The first matching is done by the matcher of ROIHeads (self.proposal_matcher).
                    self.proposal_matchers.append(None)
                else:
                    self.proposal_matchers.append(
                        Matcher([cascade_ious[k]], [0, 1], allow_low_quality_matches=False)
                    )

    def call(self, images, features, proposals, targets=None):
        image_shape = tf.shape(images.tensor)[1:3]
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        features_list = [features[f] for f in self.in_features]
        proposals = box_list.SparseBoxList.from_dense(proposals)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features_list, proposals, targets)
            if self.mask_on:
                loss_mask, deconv_features = self._forward_mask(features_list, proposals)
                proposals.data.add_field("deconv_features", deconv_features)
                losses.update(loss_mask)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances, image_shape)
            return pred_instances, {}

    def _forward_box(self, features, proposals, targets=None):
        head_outputs = []
        for k in range(self.num_cascade_stages):
            if k > 0:
                # The output boxes of the previous stage are the input proposals of the next stage
                proposals = self._create_proposals_from_boxes(
                    head_outputs[-1].predict_boxes(), proposals
                )
                if self.training:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
                proposals = box_list.SparseBoxList.from_dense(proposals)
            head_outputs.append(self._run_stage(features, proposals, k))

        if self.training:
            losses = {}
            for stage, output in enumerate(head_outputs):
                with tf.name_scope("stage{}".format(stage)):
                    stage_losses = output.losses()
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h.predict_probs() for h in head_outputs]

            # Average the scores across heads
            scores = tf.add_n(scores_per_stage) * (1.0 / self.num_cascade_stages)

            # Use the boxes of the last head
            boxes = head_outputs[-1].predict_boxes()
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                proposals,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
                self.test_nms_cls_agnostic
            )
            return pred_instances

    def _match_and_label_boxes(self, proposals, stage, targets):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.
        Args:
            proposals (BoxList): One Instances for each image, with
                the field "boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances
        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        def label_and_sample_single_image(inputs):
            proposals_dict, targets_dict = inputs

            is_valid_gt_boxes = targets_dict.pop('is_valid')
            is_crowd = tf.cast(targets_dict.pop('gt_is_crowd'), tf.bool)
            difficult = tf.cast(targets_dict.pop('gt_difficult'), tf.bool)
            valid_bool = is_valid_gt_boxes & ~is_crowd & ~difficult
            target_boxlist = box_list.BoxList.from_tensor_dict(targets_dict)
            valid_target_boxlist = box_list_ops.boolean_mask(target_boxlist, valid_bool)
            crowd_target_boxlist = box_list_ops.boolean_mask(target_boxlist, is_crowd)
            difficult_target_boxlist = box_list_ops.boolean_mask(target_boxlist, difficult)

            is_valid_proposals = proposals_dict.pop('is_valid')
            num_proposals = tf.shape(is_valid_proposals)[0]
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

            fg_inds = tf.cast(tf.where(tf.equal(proposals_labels, 1))[:, 0], tf.int32)
            bg_inds = tf.cast(tf.where(tf.equal(proposals_labels, 0))[:, 0], tf.int32)

            fg_targets = tf.gather(matched_idxs, fg_inds)
            gt_classes = tf.dynamic_stitch(
                [fg_inds, bg_inds],
                [
                    tf.gather(target_boxlist.get_field('gt_classes'), fg_targets),
                    tf.zeros_like(bg_inds, dtype=tf.int64) + self.num_classes,
                ]
            )
            gt_boxes = tf.gather(target_boxlist.boxes, matched_idxs)
            proposal_boxlist.set_field('gt_classes', gt_classes)
            proposal_boxlist.set_field('gt_boxes', gt_boxes)
            proposal_boxlist.add_field('is_valid', tf.ones_like(gt_classes, tf.bool))

            proposal_boxlist = box_list_ops.pad_or_clip_boxlist(proposal_boxlist, num_proposals)
            proposal_dict = proposal_boxlist.as_tensor_dict()
            proposal_dict['image_shape'] = image_shape
            return proposal_dict

        proposals_dict = proposals.as_tensor_dict()
        targets_dict = targets.as_tensor_dict()
        dtype = {
            field: value.dtype for field, value in proposals_dict.items()
        }
        proposals_dict = tf.map_fn(
            label_and_sample_single_image,
            [proposals_dict, targets_dict],
            dtype=dtype,
        )
        proposals = box_list.BoxList.from_tensor_dict(proposals_dict, trackings=['image_shape'])
        return proposals

    def _run_stage(self, features, proposals, stage):
        """
        Args:
            features (list[Tensor]): #lvl input features to ROIHeads
            proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
            stage (int): the current stage
        Returns:
            FastRCNNOutputs: the output of this stage
        """
        box_features = self.box_pooler(features, proposals)
        # The original implementation averages the losses among heads,
        # but scale up the parameter gradients of the heads.
        # This is equivalent to adding the losses among heads,
        # but scale down the gradients on features.
        box_features = self.scale_gradient(box_features)
        box_features = self.box_head[stage](box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor[stage](box_features)

        outputs = FastRCNNOutputs(
            self.box2box_transform[stage],
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        return outputs

    def _create_proposals_from_boxes(self, boxes, proposals):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)
        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients
        proposals.boxes = tf.stop_gradient(boxes)
        proposals = proposals.to_dense()
        proposal_dict = proposals.as_tensor_dict(trackings=['image_shape'])

        def clip_boxes_single_image(proposal_dict):
            image_shape = proposal_dict.pop('image_shape')
            proposals_per_image = box_list.BoxList.from_tensor_dict(proposal_dict)
            window = tf.cast([0, 0, image_shape[0], image_shape[1]], tf.float32)
            proposals_per_image = box_list_ops.clip_to_window(
                proposals_per_image, window, filter_nonoverlapping=False)
            if self.training:
                is_valid = tf.greater(box_list_ops.area(proposals_per_image), 0.)
                proposals_per_image.set_field('is_valid', is_valid)
            ret = proposals_per_image.as_tensor_dict()
            ret['image_shape'] = image_shape
            return ret

        clipped_proposal_dict = tf.map_fn(clip_boxes_single_image, proposal_dict)
        clipped_proposals = box_list.BoxList.from_tensor_dict(
            clipped_proposal_dict, trackings=['image_shape'])
        return clipped_proposals
