import tensorflow as tf
from tensorflow.python.training import moving_averages
import math

from ...layers import (
    Layer,
    Sequential,
    Conv2D,
    sigmoid_focal_loss,
    smooth_l1_loss
)
from ..anchor_generator import build_anchor_generator
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ...data import fields
from ...utils.shape_utils import combined_static_and_dynamic_shape
from ...utils.arg_scope import arg_scope
from ...structures import box_list, box_list_ops
from .build import SINGLE_STAGE_HEADS_REGISTRY

slim = tf.contrib.slim

__all__ = ["RetinaNetHead"]


def reshape_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, H, W, (A x K)) to (N, (HxWxA), K)
    """
    tensor.shape.assert_has_rank(4)
    N, H, W, _ = combined_static_and_dynamic_shape(tensor)
    tensor = tf.reshape(tensor, [N, -1, K]) # Size=(N,HWA,K)
    return tensor


def reshape_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, Hi, Wi, A x K)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, reshape the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [reshape_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [reshape_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = tf.concat(box_cls_flattened, axis=1)
    box_cls = tf.reshape(box_cls, [-1, num_classes])
    box_delta = tf.concat(box_delta_flattened, axis=1)
    box_delta = tf.reshape(box_delta, [-1, 4])
    return box_cls, box_delta


@SINGLE_STAGE_HEADS_REGISTRY.register()
class RetinaNetHead(Layer):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, cfg, input_shape, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = cfg.MODEL.SINGLE_STAGE_HEAD.NUM_CLASSES
        self.in_features = cfg.MODEL.SINGLE_STAGE_HEAD.IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
        # Inference parameters:
        self.score_threshold = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        feature_shapes = [input_shape[f] for f in self.in_features]
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)
        self.head = RetinaNetBoxTower(
            cfg, feature_shapes, self.anchor_generator.num_cell_anchors, scope="head"
        )

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.SINGLE_STAGE_HEAD.IOU_THRESHOLDS,
            cfg.MODEL.SINGLE_STAGE_HEAD.IOU_LABELS,
            allow_low_quality_matches=True,
        )

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        # initialize with any reasonable #fg that's not too small
        self.loss_normalizer = slim.model_variable(
            "loss_normalizer",
            shape=[],
            dtype=tf.float32,
            initializer=tf.constant_initializer(100.),
            trainable=False
        )
        self.loss_normalizer_momentum = 0.9

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
        del images
        
        features = [features[f] for f in self.in_features]
        box_cls, box_delta = self.head(features)
        anchors = self.anchor_generator(features)

        if self.training:
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(anchors, targets)
            losses = self.losses(gt_classes, gt_anchors_reg_deltas, box_cls, box_delta)
            return None, losses
        else:
            results = self.inference(box_cls, box_delta, anchors)
            return results, {}

    def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas = reshape_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = tf.reshape(gt_classes, [-1])
        gt_anchors_deltas = tf.reshape(gt_anchors_deltas, [-1, 4])

        valid_idxs = tf.where(tf.greater_equal(gt_classes, 0))[:, 0]
        foreground_idxs = tf.where(
            tf.logical_and(
                tf.greater_equal(gt_classes, 0), tf.less(gt_classes, self.num_classes)
            )
        )[:, 0]
        
        sparse_gt_classes_target = tf.sparse.SparseTensor(
            tf.stack([foreground_idxs, tf.gather(gt_classes, foreground_idxs)], axis=1),
            tf.ones_like(foreground_idxs, dtype=tf.float32),
            tf.shape(pred_class_logits, out_type=tf.int64)
        )
        gt_classes_target = tf.sparse.to_dense(sparse_gt_classes_target)

        # logits loss
        loss_cls = sigmoid_focal_loss(
            predictions=tf.gather(pred_class_logits, valid_idxs),
            targets=tf.gather(gt_classes_target, valid_idxs),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        # regression loss
        loss_box_reg = smooth_l1_loss(
            predictions=tf.gather(pred_anchor_deltas, foreground_idxs),
            labels=tf.gather(gt_anchors_deltas, foreground_idxs),
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        )

        num_foreground = tf.cast(tf.shape(foreground_idxs)[0], tf.float32)
        tf.summary.scalar("retinanet/num_foreground", num_foreground)
        update_loss_normalizer = moving_averages.assign_moving_average(
            self.loss_normalizer, tf.maximum(1., num_foreground),
            self.loss_normalizer_momentum, zero_debias=False)
        with tf.control_dependencies([update_loss_normalizer]):
            loss_cls = loss_cls / self.loss_normalizer
            loss_box_reg = loss_box_reg / self.loss_normalizer
        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    def get_ground_truth(self, anchors, targets):
        """
        Args:
            anchors (list[BoxList]): A list of N elements. Each element is a BoxList 
                storing the entire anchor array for feature map l.
            gt_boxes (BoxList): storing the ground-truth ("gt") boxes.
        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
                Anchors with an IoU with some target higher than the foreground threshold
                are assigned their corresponding label in the [0, K-1] range.
                Anchors whose IoU are below the background threshold are assigned
                the label "K". Anchors whose IoU are between the foreground and background
                thresholds are assigned a label "-1", i.e. ignore.
            gt_anchors_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth box2box transform
                targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground.
        """
        anchor_boxlist = box_list_ops.concatenate(anchors)

        def get_ground_truth_single_image(gt_box_dict_i):
            is_valid = gt_box_dict_i.pop('is_valid')
            gt_boxlist_i = box_list.BoxList.from_tensor_dict(gt_box_dict_i)
            gt_boxlist_i = box_list_ops.boolean_mask(gt_boxlist_i, is_valid)

            match_quality_matrix = box_list_ops.pairwise_iou(gt_boxlist_i, anchor_boxlist)
            matches, gt_labels_i = self.matcher(match_quality_matrix)

            positive_inds = tf.cast(
                tf.reshape(tf.where(tf.greater(gt_labels_i, 0)), [-1]), tf.int32)
            positive_matches = tf.gather(matches, positive_inds)
            positive_matched_anchors = box_list_ops.gather(anchor_boxlist, positive_inds)
            positive_matched_gt_boxes = box_list_ops.gather(gt_boxlist_i, positive_matches)
            positive_deltas = self.box2box_transform.get_deltas(
                positive_matched_anchors.boxes, positive_matched_gt_boxes.boxes
            )
            negative_inds = tf.cast(
                tf.reshape(tf.where(tf.equal(gt_labels_i, 0)), [-1]), tf.int32)
            ignored_inds = tf.cast(
                tf.reshape(tf.where(tf.equal(gt_labels_i, -1)), [-1]), tf.int32)

            gt_anchor_deltas_i = tf.dynamic_stitch(
                [positive_inds, negative_inds, ignored_inds],
                [
                    positive_deltas,
                    tf.zeros([tf.shape(negative_inds)[0], 4], dtype=tf.float32),
                    tf.zeros([tf.shape(ignored_inds)[0], 4], dtype=tf.float32)
                ]
            )
            gt_classes_i = tf.dynamic_stitch(
                [positive_inds, negative_inds, ignored_inds],
                [
                    positive_matched_gt_boxes.get_field("gt_classes"),
                    tf.zeros([tf.shape(negative_inds)[0]], dtype=tf.int64) + self.num_classes,
                    tf.zeros([tf.shape(ignored_inds)[0]], dtype=tf.int64) - 1
                ]
            )

            return gt_classes_i, gt_anchor_deltas_i

        gt_box_dict = targets.as_tensor_dict()
        gt_classes, gt_anchor_deltas = tf.map_fn(
            get_ground_truth_single_image,
            gt_box_dict,
            dtype=(tf.int64, tf.float32),
            back_prop=False
        )

        return gt_classes, gt_anchor_deltas

    def inference(self, box_cls, box_delta, anchors):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            image_shapes ([N, 2]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """

        result_fields = fields.ResultFields

        def inference_single_image(args):
            """
            Single-image inference. Return bounding-box detection results by thresholding
            on scores and applying non-maximum suppression (NMS).
            Arguments:
                box_cls (list[Tensor]): list of #feature levels. Each entry contains
                    tensor of size (H x W x A, K)
                box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
                anchors (list[Boxes]): list of #feature levels. Each entry contains
                    a Boxes object, which contains all the anchors for that
                    image in that feature level.
            Returns:
                Same as `inference`, but for only one image.
            """
            box_cls, box_delta = args
            boxes_all = []
            scores_all = []
            class_idxs_all = []

            # Iterate over every feature level
            for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
                # (HxWxAxK,)
                box_cls_i = tf.reshape(box_cls_i, [-1])
                box_cls_i = tf.nn.sigmoid(box_cls_i)

                # Keep top k top scoring indices only.
                num_topk = tf.minimum(self.topk_candidates, tf.shape(box_reg_i)[0])
                predicted_prob, topk_idxs = tf.nn.top_k(box_cls_i, k=num_topk, sorted=True)

                # filter out the proposals with low confidence score
                keep_idxs = tf.where(predicted_prob > self.score_threshold)[:, 0]
                predicted_prob = tf.gather(predicted_prob, keep_idxs)
                topk_idxs = tf.gather(topk_idxs, keep_idxs)

                anchor_idxs = topk_idxs // self.num_classes
                classes_idxs = topk_idxs % self.num_classes

                box_reg_i = tf.gather(box_reg_i, anchor_idxs)
                anchors_i = box_list_ops.gather(anchors_i, anchor_idxs)
                # predict boxes
                predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.boxes)

                boxes_all.append(predicted_boxes)
                scores_all.append(predicted_prob)
                class_idxs_all.append(classes_idxs)

            boxes_all, scores_all, class_idxs_all = [
                tf.concat(x, axis=0) for x in [boxes_all, scores_all, class_idxs_all]
            ]

            max_coord = tf.reduce_max(boxes_all)
            offsets = tf.cast(class_idxs_all, tf.float32) * (max_coord + 1)  # R'
            boxes_nms = boxes_all + tf.expand_dims(offsets, axis=-1)

            keep = tf.image.non_max_suppression(
                boxes_nms, scores_all, self.max_detections_per_image, self.nms_threshold
            )
            boxes = tf.gather(boxes_all, keep, name="boxes")
            scores = tf.gather(scores_all, keep, name="scores")
            class_idxs = tf.gather(class_idxs_all, keep, name="classes")

            result_boxlist = box_list.BoxList(boxes)
            result_boxlist.add_field('scores', scores)
            result_boxlist.add_field('pred_classes', class_idxs)
            result_boxlist.add_field('is_valid', tf.ones_like(scores, dtype=tf.bool))
            # pad to topk_per_image in case num nmsed boxes < topk_per_image
            result_boxlist = box_list_ops.pad_or_clip_boxlist(
                result_boxlist, self.max_detections_per_image
            )

            return result_boxlist.as_tensor_dict()

        box_cls = [reshape_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [reshape_to_N_HWA_K(x, 4) for x in box_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        result_boxlist_dict = tf.map_fn(
            inference_single_image,
            [box_cls, box_delta],
            dtype={
                'boxes': tf.float32,
                'scores': tf.float32,
                'pred_classes': tf.int32,
                'is_valid': tf.bool,
            }
        )

        result_boxlist = box_list.BoxList.from_tensor_dict(result_boxlist_dict)
        return result_boxlist


class RetinaNetBoxTower(Layer):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape, num_anchors, **kwargs):
        super().__init__(**kwargs)
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.SINGLE_STAGE_HEAD.NUM_CLASSES
        num_convs = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            cls_subnet = []
            bbox_subnet = []
            with arg_scope(
                [Conv2D],
                kernel_size=3,
                stride=1,
                padding='SAME',
                activation=tf.nn.relu,
                weights_initializer=tf.random_normal_initializer(stddev=0.01),
            ):
                for i in range(num_convs):
                    cls_subnet.append(Conv2D(in_channels, in_channels, scope=f"cls_subnet{2*i}"))
                    bbox_subnet.append(Conv2D(in_channels, in_channels, scope=f"bbox_subnet{2*i}"))

                self.cls_subnet = Sequential(cls_subnet)
                self.bbox_subnet = Sequential(bbox_subnet)
                # Use prior in model initialization to improve stability
                bias_initializer = tf.constant_initializer(-math.log((1 - prior_prob) / prior_prob))
                self.cls_score = Conv2D(
                    in_channels, num_anchors * num_classes, activation=None,
                    bias_initializer=bias_initializer, scope="cls_score")
                self.bbox_pred = Conv2D(in_channels, num_anchors * 4, activation=None, scope="bbox_pred")

    def call(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.
        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, Hi, Wi, AxK).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Hi, Wi, Ax4).
                The tensor predicts 4-vector (dy, dx, dh, dw) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg