import itertools
import tensorflow as tf

from ...layers import smooth_l1_loss, flatten
from ...structures import box_list
from ...structures import box_list_ops
from ...utils import shape_utils
from ..sampling import subsample_labels

slim = tf.contrib.slim

"""
Shape shorthand in this module:
    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization
Naming convention:
    objectness: refers to the binary classification of an anchor as object vs. not
    object.
    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).
    pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
        sigmoid(pred_objectness_logits) to estimate P(object).
"""


def find_top_rpn_proposals(
    proposals,
    pred_objectness_logits,
    images,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_side_len,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    , clip proposals, remove small boxes, and apply NMS.
    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        images (ImageList): Input images as an :class:`ImageList`.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_side_len (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
    Returns:
        proposals (BoxList): representing list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i.
    """

    def find_top_rpn_proposals_single_image(inputs):
        proposals, pred_objectness_logits, image_shape = inputs

        proposal_boxes = []
        proposal_scores = []
        for proposals_i, logits_i in zip(proposals, pred_objectness_logits):
            # 1. Select top-k anchor
            Hi_Wi_A = tf.shape(logits_i)[0]
            topk_i = tf.minimum(pre_nms_topk, Hi_Wi_A)

            topk_scores_i, topk_idx = tf.nn.top_k(logits_i, k=topk_i, sorted=False)
            topk_proposals_i = tf.gather(proposals_i, topk_idx)

            # 2. clip proposals
            boxlist_i = box_list.BoxList(topk_proposals_i)
            boxlist_i.add_field("scores", topk_scores_i)

            window = tf.stack([0, 0, image_shape[0], image_shape[1]])
            window = tf.cast(window, tf.float32)
            clipped_boxlist_i = box_list_ops.clip_to_window(
                boxlist_i, window, filter_nonoverlapping=False)

            # 3. prune small boxes
            if min_box_side_len > 0:
                valid_boxlist_i = box_list_ops.prune_small_boxes(
                    clipped_boxlist_i, min_box_side_len)
            else:
                valid_boxlist_i = clipped_boxlist_i

            # 4. nms
            keep = tf.image.non_max_suppression(
                valid_boxlist_i.boxes,
                valid_boxlist_i.get_field("scores"),
                max_output_size=post_nms_topk,
                iou_threshold=nms_thresh)
            nmsed_boxlist_i = box_list_ops.gather(
                valid_boxlist_i, keep)

            proposal_boxes.append(nmsed_boxlist_i.boxes)
            proposal_scores.append(nmsed_boxlist_i.get_field("scores"))

        proposal_boxes = tf.concat(proposal_boxes, axis=0)
        proposal_scores = tf.concat(proposal_scores, axis=0)

        # 5. find topk proposals in an image
        proposal_topk = tf.minimum(tf.size(proposal_scores), post_nms_topk)
        proposal_scores, topk_indices = tf.nn.top_k(proposal_scores, k=proposal_topk, sorted=True)
        proposal_boxes = tf.gather(proposal_boxes, topk_indices)
        proposal_is_valid = tf.ones_like(proposal_scores, tf.bool)

        # 6. pad to the same length
        P = post_nms_topk - proposal_topk
        proposal_boxes = tf.pad(proposal_boxes, [[0, P], [0, 0]])
        proposal_scores = tf.pad(proposal_scores, [[0, P]])
        proposal_is_valid = tf.pad(proposal_is_valid, [[0, P]])

        proposal_boxes = tf.stop_gradient(proposal_boxes, name='boxes')
        proposal_scores = tf.stop_gradient(proposal_scores, name='logits')
        proposal_is_valid = tf.stop_gradient(proposal_is_valid)
        return proposal_boxes, proposal_scores, proposal_is_valid

    image_shapes = images.image_shapes  # [B,2] in (h, w) order

    proposal_boxes, proposal_scores, proposal_is_valid = tf.map_fn(
        find_top_rpn_proposals_single_image,
        (proposals, pred_objectness_logits, image_shapes),
        dtype=(tf.float32, tf.float32, tf.bool))

    results = box_list.BoxList(proposal_boxes)
    results.add_field("objectness_logits", proposal_scores)
    results.add_field("is_valid", proposal_is_valid)
    results.set_tracking("image_shape", image_shapes)
    return results


def rpn_losses(
    gt_objectness_logits,
    gt_anchor_deltas,
    pred_objectness_logits,
    pred_anchor_deltas,
    smooth_l1_beta,
):
    """
    Args:
        gt_objectness_logits (Tensor): shape (N,), each element in {-1, 0, 1} representing
            ground-truth objectness labels with: -1 = ignore; 0 = not object; 1 = object.
        gt_anchor_deltas (Tensor): shape (N, box_dim), row i represents ground-truth
            box2box transform targets (dx, dy, dw, dh) or (dx, dy, dw, dh, da) that map anchor i to
            its matched ground-truth box.
        pred_objectness_logits (Tensor): shape (N,), each element is a predicted objectness
            logit.
        pred_anchor_deltas (Tensor): shape (N, box_dim), each row is a predicted box2box
            transform (dx, dy, dw, dh) or (dx, dy, dw, dh, da)
        smooth_l1_beta (float): The transition point between L1 and L2 loss in
            the smooth L1 loss function. When set to 0, the loss becomes L1. When
            set to +inf, the loss becomes constant 0.
    Returns:
        objectness_loss, localization_loss, both unnormalized (summed over samples).
    """
    pos_masks = tf.stop_gradient(tf.equal(gt_objectness_logits, 1))
    localization_loss = tf.cond(
        tf.reduce_any(pos_masks),
        lambda: smooth_l1_loss(
            labels=tf.boolean_mask(gt_anchor_deltas, pos_masks),
            predictions=tf.boolean_mask(pred_anchor_deltas, pos_masks),
            beta=smooth_l1_beta,
            reduction='sum',
        ),
        lambda: 0.0
    )

    valid_masks = tf.stop_gradient(tf.greater_equal(gt_objectness_logits, 0))
    logits = tf.boolean_mask(pred_objectness_logits, valid_masks)
    labels = tf.cast(
        tf.boolean_mask(gt_objectness_logits, valid_masks), tf.float32
    )
    objectness_loss = tf.cond(
        tf.reduce_any(valid_masks),
        lambda: tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels
            )
        ),
        lambda: 0.0
    )
    return objectness_loss, localization_loss


class RPNOutputs(object):
    def __init__(
        self,
        box2box_transform,
        anchor_matcher,
        batch_size_per_image,
        positive_fraction,
        images,
        pred_objectness_logits,
        pred_anchor_deltas,
        anchors,
        boundary_threshold=0,
        gt_boxes=None,
        smooth_l1_beta=0.0,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for anchors.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi, Wi, A*4) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[BoxList]): A list of N elements. Each element is a list of L
                Boxes. The Boxes at (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training. Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_boxes (BoxList, optional): storing the ground-truth ("gt") boxes.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pred_objectness_logits = pred_objectness_logits
        self.pred_anchor_deltas = pred_anchor_deltas

        self.anchors = anchors
        self.gt_boxes = gt_boxes
        self.num_feature_maps = len(pred_objectness_logits)
        self.num_images = images.num_images
        if self.gt_boxes is not None:
            self.gt_boxes.set_tracking('image_shape', images.image_shapes)
        self.boundary_threshold = boundary_threshold
        self.smooth_l1_beta = smooth_l1_beta

    def _get_ground_truth(self):
        """
        Returns:
            gt_objectness_logits: [N, sum(Hi*Wi*A)] tensors. . Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: [N, sum(Hi*Wi*A), 4].
        """
        # Concatenate anchors from all feature maps into a single Boxes per image
        anchor_boxlist = box_list_ops.concatenate(self.anchors)

        def get_ground_truth_single_image(gt_box_dict_i):
            image_shape_i = gt_box_dict_i.pop('image_shape')
            is_valid = gt_box_dict_i.pop('is_valid')
            is_crowd = tf.cast(gt_box_dict_i.pop('gt_is_crowd'), tf.bool)
            is_valid_bool = is_valid & ~is_crowd
            gt_boxlist_i = box_list.BoxList.from_tensor_dict(gt_box_dict_i)
            valid_gt_boxlist_i = box_list_ops.boolean_mask(gt_boxlist_i, is_valid_bool)
            crowd_gt_boxlist_i = box_list_ops.boolean_mask(gt_boxlist_i, is_crowd)

            match_quality_matrix = box_list_ops.pairwise_iou(valid_gt_boxlist_i, anchor_boxlist)
            crowd_matrix = box_list_ops.pairwise_iou(crowd_gt_boxlist_i, anchor_boxlist)
            matches, gt_objectness_logits_i = self.anchor_matcher(
                match_quality_matrix, crowd_matrix)

            if self.boundary_threshold >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default
                window_i = tf.stack([0, 0, image_shape_i[0], image_shape_i[1]])
                window_i = tf.cast(window_i, tf.float32)
                inside_masks = box_list_ops.inside_window(
                    anchor_boxlist, window_i, self.boundary_threshold)
                gt_objectness_logits_i = tf.where(
                    inside_masks,
                    gt_objectness_logits_i,
                    tf.zeros_like(gt_objectness_logits_i) - 1)

            positive_inds = tf.cast(tf.where(gt_objectness_logits_i > 0)[:, 0], tf.int32)
            positive_matches = tf.gather(matches, positive_inds)
            positive_matched_anchors = box_list_ops.gather(anchor_boxlist, positive_inds)
            positive_matched_gt_boxes = box_list_ops.gather(valid_gt_boxlist_i, positive_matches)
            positive_deltas = self.box2box_transform.get_deltas(
                positive_matched_anchors.boxes, positive_matched_gt_boxes.boxes
            )
            all_inds = tf.range(tf.shape(gt_objectness_logits_i)[0], dtype=tf.int32)
            gt_anchor_deltas_i = tf.dynamic_stitch(
                [all_inds, positive_inds],
                [tf.zeros_like(anchor_boxlist.boxes), positive_deltas]
            )

            return gt_objectness_logits_i, gt_anchor_deltas_i

        gt_box_dict = self.gt_boxes.as_tensor_dict(trackings=['image_shape'])
        gt_objectness_logits, gt_anchor_deltas = tf.map_fn(
            get_ground_truth_single_image,
            gt_box_dict,
            dtype=(tf.int64, tf.float32),
            back_prop=False
        )

        return gt_objectness_logits, gt_anchor_deltas

    def losses(self):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.
        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """

        def resample(label):
            """
            Randomly sample a subset of positive and negative examples by overwriting
            the label vector to the ignore value (-1) for all elements that are not
            included in the sample.
            """
            pos_idx, neg_idx = subsample_labels(
                label, self.batch_size_per_image, self.positive_fraction, 0
            )
            # Fill with the ignore label (-1), then set positive and negative labels
            all_idx = tf.range(tf.shape(label)[0])
            label = tf.dynamic_stitch(
                [all_idx, tf.cast(pos_idx, tf.int32), tf.cast(neg_idx, tf.int32)],
                [tf.zeros_like(label) - 1, tf.ones_like(pos_idx), tf.zeros_like(neg_idx)]
            )
            return label

        gt_objectness_logits, gt_anchor_deltas = self._get_ground_truth()

        # resample: (N, num_anchors_per_image)
        gt_objectness_logits = tf.map_fn(resample, gt_objectness_logits)

        # Collect all objectness labels and delta targets over feature maps and images
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        num_anchors_per_map = [tf.reduce_prod(tf.shape(x)[1:]) for x in self.pred_objectness_logits]
        num_anchors_per_image = tf.reduce_sum(num_anchors_per_map)

        # Log the number of positive/negative anchors per-image that's used in training
        num_pos_anchors = tf.count_nonzero(
            tf.equal(gt_objectness_logits, 1), dtype=tf.int32)
        num_neg_anchors = tf.count_nonzero(
            tf.equal(gt_objectness_logits, 0), dtype=tf.int32)
        tf.summary.scalar("rpn/num_pos_anchors", num_pos_anchors / self.num_images)
        tf.summary.scalar("rpn/num_neg_anchors", num_neg_anchors / self.num_images)

        with tf.control_dependencies(
                [tf.assert_equal(tf.shape(gt_objectness_logits)[1], num_anchors_per_image)]):
            # Split to tuple of L tensors, each with shape (N, num_anchors_per_map)
            gt_objectness_logits = tf.split(
                gt_objectness_logits, num_anchors_per_map, axis=1)
            # Concat from all feature maps
            gt_objectness_logits = tf.concat(
                [tf.reshape(x, [-1]) for x in gt_objectness_logits], axis=0)

        gt_anchor_deltas_shape = shape_utils.combined_static_and_dynamic_shape(gt_anchor_deltas)
        with tf.control_dependencies(
                [tf.assert_equal(gt_anchor_deltas_shape[1], num_anchors_per_image)]):
            B = gt_anchor_deltas_shape[2]  # box dimension (4 or 5)

            # Split to tuple of L tensors, each with shape (N, num_anchors_per_image)
            gt_anchor_deltas = tf.split(gt_anchor_deltas, num_anchors_per_map, axis=1)
            # Concat from all feature maps
            gt_anchor_deltas = tf.concat(
                [tf.reshape(x, [-1, B]) for x in gt_anchor_deltas], axis=0)

        # Collect all objectness logits and delta predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        pred_objectness_logits = tf.concat(
            [
                # Reshape: (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
                tf.reshape(x, [-1]) for x in self.pred_objectness_logits
            ],
            axis=0,
        )
        pred_anchor_deltas = tf.concat(
            [
                # Reshape: (N, Hi, Wi, A*B) -> (N*Hi*Wi*A, B)
                tf.reshape(x, [-1, B])
                for x in self.pred_anchor_deltas
            ],
            axis=0,
        )

        objectness_loss, localization_loss = rpn_losses(
            gt_objectness_logits,
            gt_anchor_deltas,
            pred_objectness_logits,
            pred_anchor_deltas,
            self.smooth_l1_beta,
        )
        normalizer = 1.0 / tf.cast(self.batch_size_per_image * self.num_images, tf.float32)
        loss_cls = objectness_loss * normalizer  # cls: classification loss
        loss_loc = localization_loss * normalizer  # loc: localization loss
        losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc}

        return losses

    def predict_proposals(self):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.
        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(self.anchors, self.pred_anchor_deltas):
            B = shape_utils.combined_static_and_dynamic_shape(anchors_i.boxes)[1]
            N, Hi, Wi, _ = shape_utils.combined_static_and_dynamic_shape(pred_anchor_deltas_i)
            # Reshape: (N, Hi, Wi, A*B) ->  (N*Hi*Wi*A, B)
            pred_anchor_deltas_i = tf.reshape(pred_anchor_deltas_i, [-1, B])
            # tile anchors to shape (N*Hi*Wi*A, B)
            anchors_i = tf.tile(
                tf.expand_dims(anchors_i.boxes, axis=0), [self.num_images, 1, 1])
            anchors_i = tf.reshape(anchors_i, [-1, B])
            proposals_i = self.box2box_transform.apply_deltas(
                pred_anchor_deltas_i, anchors_i
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(tf.reshape(proposals_i, [N, -1, B]))
        return proposals

    def predict_objectness_logits(self):
        """
        Return objectness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.
        Returns:
            pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A).
        """
        pred_objectness_logits = [
            # Reshape: (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            flatten(score) for score in self.pred_objectness_logits
        ]
        return pred_objectness_logits
