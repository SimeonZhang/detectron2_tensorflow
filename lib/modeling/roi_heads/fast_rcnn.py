import tensorflow as tf
import numpy as np

from ...structures import box_list
from ...structures import box_list_ops
from ...utils import shape_utils
from ...layers import Layer, Linear, flatten, smooth_l1_loss

"""
Shape shorthand in this module:
    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.
Naming convention:
    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).
    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).
    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.
    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.
    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(
    boxes,
    scores,
    proposals,
    score_thresh,
    nms_thresh,
    topk_per_image,
    nms_cls_agnostic
):
    """
    Postprocess predicted boxes.
    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.
    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    with tf.name_scope("FastRCNNPostprocess"):
        num_bbox_reg_classes = (
            shape_utils.combined_static_and_dynamic_shape(boxes)[1] // 4
        )

        # shape of field `scores` should be compatible with boxes
        scores = scores[:, :-1]
        num_preds, num_classes = shape_utils.combined_static_and_dynamic_shape(scores)
        pred_indices = proposals.indices
        pred_dense_shape = proposals.dense_shape

        if num_bbox_reg_classes != 1:
            boxes = tf.reshape(
                boxes, [num_preds * num_bbox_reg_classes, 4]
            )
            scores = tf.reshape(
                scores, [-1]
            )
            # construct the predicted InstanceList
            orig_masks = tf.sparse.SparseTensor(
                proposals.indices,
                tf.ones([tf.shape(proposals.indices)[0]], dtype=tf.bool),
                proposals.dense_shape
            )
            orig_masks = tf.sparse.to_dense(orig_masks, default_value=False)
            pred_dense_shape = [
                proposals.dense_shape[0],
                proposals.dense_shape[1] * num_bbox_reg_classes
            ]
            pred_masks = tf.reshape(
                tf.tile(
                    tf.expand_dims(orig_masks, axis=2),
                    [1, 1, num_bbox_reg_classes]
                ),
                pred_dense_shape
            )
            pred_indices = tf.where(pred_masks)

        pred_boxlist = box_list.BoxList(boxes)
        pred_boxlist.add_field('scores', scores)
        sparse_pred_boxlist = box_list.SparseBoxList(
            pred_indices,
            pred_boxlist,
            pred_dense_shape,
        )
        sparse_pred_boxlist.set_tracking('image_shape', proposals.get_tracking('image_shape'))
        pred_boxlist = sparse_pred_boxlist.to_dense()

        def fast_rcnn_inference_single_image(pred_boxlist_dict):
            # 1. clip
            boxlist = box_list.BoxList(pred_boxlist_dict['boxes'])
            image_shape = pred_boxlist_dict['image_shape']
            window = tf.stack([0, 0, image_shape[0], image_shape[1]])
            window = tf.cast(window, tf.float32)
            clipped_boxlist = box_list_ops.clip_to_window(
                boxlist, window, filter_nonoverlapping=False
            )
            boxes = tf.reshape(clipped_boxlist.boxes, [-1, num_bbox_reg_classes, 4])
            scores = tf.reshape(pred_boxlist_dict['scores'], [-1, num_classes])
            boxes = tf.transpose(boxes, [1, 0, 2])
            scores = tf.transpose(scores, [1, 0])

            # 2. Filter results based on detection scores
            filtered_mask = scores > score_thresh  # R x K
            # R' x 2. First column contains indices of the R predictions;
            # Second column contains indices of classes.
            filtered_inds = tf.where(filtered_mask)
            cls_per_box = tf.slice(filtered_inds, [0, 0], [-1, 1])
            ind_per_box = tf.slice(filtered_inds, [0, 1], [-1, -1])

            # 3. Apply per-class NMS
            if num_bbox_reg_classes == 1:
                filtered_box_inds = tf.concat(
                    [tf.zeros_like(ind_per_box), ind_per_box], axis=1)
                filtered_boxes = tf.gather_nd(boxes, filtered_box_inds)  # R' x 4
            else:
                filtered_boxes = tf.gather_nd(boxes, filtered_inds)
            filtered_scores = tf.gather_nd(scores, filtered_inds)  # R',
            if nms_cls_agnostic:
                nms_boxes = filtered_boxes
            else:
                max_coord = tf.reduce_max(boxes)
                offsets = tf.cast(cls_per_box, tf.float32) * (max_coord + 1)  # R' x 1
                nms_boxes = filtered_boxes + offsets

            keep = tf.image.non_max_suppression(
                nms_boxes, filtered_scores, topk_per_image, nms_thresh)
            boxes = tf.gather(filtered_boxes, keep, name="boxes")
            scores = tf.gather(filtered_scores, keep, name="scores")
            filtered_inds = tf.gather(filtered_inds, keep)

            result_boxlist = box_list.BoxList(boxes)
            result_boxlist.add_field('scores', scores)
            result_boxlist.add_field('pred_classes', filtered_inds[:, 0])
            result_boxlist.add_field('is_valid', tf.ones_like(scores, dtype=tf.bool))

            # pad to topk_per_image in case num nmsed boxes < topk_per_image
            result_boxlist = box_list_ops.pad_or_clip_boxlist(
                result_boxlist, topk_per_image
            )
            num_valid = tf.shape(filtered_inds)[0]
            kept_mask = tf.sparse.SparseTensor(
                filtered_inds[:, 1:2],
                tf.ones([tf.shape(filtered_inds)[0]], dtype=tf.bool),
                [num_preds]
            )
            kept_mask = tf.sparse.to_dense(kept_mask, default_value=False)
            result_boxlist_dict = result_boxlist.as_tensor_dict()
            return result_boxlist_dict, kept_mask

        pred_boxlist_dict = pred_boxlist.as_tensor_dict()
        result_boxlist_dict, kept_masks = tf.map_fn(
            fast_rcnn_inference_single_image,
            pred_boxlist_dict,
            dtype=(
                {
                    'boxes': tf.float32,
                    'scores': tf.float32,
                    'pred_classes': tf.int64,
                    'is_valid': tf.bool,
                },
                tf.bool
            ),
        )
        result_boxlist = box_list.BoxList.from_tensor_dict(result_boxlist_dict)
        result_boxlist.set_tracking('image_shape', pred_boxlist.get_tracking('image_shape'))
        kept_indices = tf.where(kept_masks)
        return result_boxlist, kept_indices


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (SparseBoxList): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.proposals = proposals
        self.smooth_l1_beta = smooth_l1_beta

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        gt_classes = self.proposals.data.get_field('gt_classes')
        num_instances = tf.shape(gt_classes)[0]
        pred_classes = tf.argmax(self.pred_class_logits, axis=1, output_type=tf.int64)
        bg_class_ind = shape_utils.combined_static_and_dynamic_shape(self.pred_class_logits)[1] - 1

        fg_inds = tf.where((gt_classes >= 0) & (gt_classes < bg_class_ind))
        num_fg = tf.shape(fg_inds)[0]
        fg_gt_classes = tf.gather_nd(gt_classes, fg_inds)
        fg_pred_classes = tf.gather_nd(pred_classes, fg_inds)

        num_false_negative = tf.count_nonzero(
            tf.equal(fg_pred_classes, bg_class_ind), dtype=tf.int32)
        num_accurate = tf.count_nonzero(
            tf.equal(pred_classes, gt_classes), dtype=tf.int32)
        fg_num_accurate = tf.count_nonzero(
            tf.equal(fg_pred_classes, fg_gt_classes), dtype=tf.int32)

        tf.summary.scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
        tf.summary.scalar(
            "fast_rcnn/fg_cls_accuracy",
            tf.cond(
                num_fg > 0,
                lambda: tf.cast(fg_num_accurate / num_fg, tf.float32),
                lambda: 0.
            )
        )
        tf.summary.scalar(
            "fast_rcnn/false_negative",
            tf.cond(
                num_fg > 0,
                lambda: tf.cast(num_false_negative / num_fg, tf.float32),
                lambda: 0.
            )
        )

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        loss_cls = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.proposals.data.get_field('gt_classes'),
                logits=self.pred_class_logits
            )
        )
        loss_cls = tf.cond(
            tf.shape(self.pred_class_logits)[0] > 0, lambda: loss_cls, lambda: 0.
        )
        return loss_cls

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.
        Returns:
            scalar Tensor
        """
        box_dim = (
            shape_utils.combined_static_and_dynamic_shape(self.proposals.data.boxes)[-1]
        )  # 4 or 5
        num_bbox_reg_classes = (
            shape_utils.combined_static_and_dynamic_shape(self.pred_proposal_deltas)[1] // box_dim
        )
        pred_proposal_deltas = tf.reshape(
            self.pred_proposal_deltas, [-1, num_bbox_reg_classes, box_dim]
        )

        bg_class_ind = shape_utils.combined_static_and_dynamic_shape(self.pred_class_logits)[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        fg_mask = tf.logical_and(
            tf.greater_equal(self.proposals.data.get_field('gt_classes'), 0),
            tf.less(self.proposals.data.get_field('gt_classes'), bg_class_ind)
        )
        fg_inds = tf.where(fg_mask)
        fg_proposals = box_list_ops.gather(self.proposals.data, fg_inds[:, 0])
        gt_fg_proposal_deltas = self.box2box_transform.get_deltas(
            fg_proposals.boxes, fg_proposals.get_field('gt_boxes')
        )

        if num_bbox_reg_classes == 1:
            fg_classes = tf.zeros_like(fg_inds)
        else:
            fg_classes = tf.gather(self.proposals.data.get_field('gt_classes'), fg_inds)
        pred_fg_inds = tf.concat([fg_inds, fg_classes], axis=1)

        loss_box_reg = smooth_l1_loss(
            labels=gt_fg_proposal_deltas,
            predictions=tf.gather_nd(pred_proposal_deltas, pred_fg_inds),
            beta=self.smooth_l1_beta,
            reduction='sum',
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        num_preds = tf.cast(tf.shape(self.proposals.data.get_field('gt_classes'))[0], tf.float32)
        loss_box_reg = tf.cond(
            tf.reduce_any(fg_mask), lambda: loss_box_reg / num_preds, lambda: 0.
        )
        return loss_box_reg

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.
        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

    def predict_boxes(self):
        """
        Returns:
            Tensor: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas, self.proposals.data.boxes
        )
        return boxes

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = tf.nn.softmax(self.pred_class_logits)
        return probs

    def inference(self, score_thresh, nms_thresh, topk_per_image, nms_cls_agnostic):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        return fast_rcnn_inference(
            boxes, scores, self.proposals, score_thresh, nms_thresh, topk_per_image, nms_cls_agnostic
        )


class FastRCNNOutputLayers(Layer):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4, **kwargs):
        """
        Args:
            input_size (int): channels, or (height, width, channels)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular YXYX boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__(**kwargs)

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            self.cls_score = Linear(input_size, num_classes + 1,
                                    weights_initializer=tf.random_normal_initializer(stddev=0.01),
                                    scope='class_logits')
            num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
            self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim,
                                    weights_initializer=tf.random_normal_initializer(stddev=0.001),
                                    scope='box_deltas')

    def call(self, x):
        if x.shape.ndims > 2:
            x = flatten(x)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas
