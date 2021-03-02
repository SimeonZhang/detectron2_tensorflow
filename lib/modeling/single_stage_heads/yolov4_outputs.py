import tensorflow as tf

from ...layers import sigmoid_focal_loss, iou_loss
from ...utils.shape_utils import combined_static_and_dynamic_shape
from ...structures import box_list, box_list_ops
from ...data import fields


class YOLOV4Outputs(object):
    def __init__(
        self,
        matcher,
        anchor_generator,
        pred_logits,
        num_classes,
        cls_normalizer,
        iou_normalizer,
        scale_yx,
        score_threshold,
        nms_threshold,
        max_detections_per_image,
        images,
        gt_boxes=None
    ):
        """
        Args:
            matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi, Wi, A * (C + 5)) representing
                the predicted logits for anchors.
            anchors (list[BoxList]): A list of N elements. Each element is a list of L
                Boxes. The Boxes at (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            gt_boxes (BoxList, optional): storing the ground-truth ("gt") boxes.
        """
        self.matcher = matcher
        self.anchor_generator = anchor_generator
        self.pred_logits = pred_logits
        self.num_classes = num_classes
        self.gt_boxes = gt_boxes
        self.num_images = images.num_images
        self.image_shape = combined_static_and_dynamic_shape(images.tensor)[1:3]
        if self.gt_boxes is not None:
            self.gt_boxes.set_tracking('image_shape', images.image_shapes)

        self.grid_sizes = [tf.shape(logits)[1:3] for logits in pred_logits]

        self.cls_normalizer = cls_normalizer
        self.iou_normalizer = iou_normalizer
        self.scale_yx = scale_yx
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.post_nms_topk = max_detections_per_image

    def _get_ground_truth(self, predicted_boxes):
        """
        Returns:
            gt_objectness_logits: [N, sum(Hi*Wi*A)] tensors. . Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: [N, sum(Hi*Wi*A), 4].
        """
        # Concatenate anchors from all feature maps into a single Boxes per image
        anchor_boxlist = box_list_ops.concatenate(self.anchor_generator(self.pred_logits))
        num_cell_anchors = self.anchor_generator.num_cell_anchors
        assert len(set(num_cell_anchors)) == 1, num_cell_anchors
        num_cell_anchors = num_cell_anchors[0]
        cell_anchor_list = self.anchor_generator.cell_anchors
        strides = self.anchor_generator.strides

        def get_ground_truth_single_image(args):
            gt_box_dict_i, predicted_boxes = args
            image_shape_i = gt_box_dict_i.pop('image_shape')
            is_valid = gt_box_dict_i.pop('is_valid')
            is_crowd = tf.cast(gt_box_dict_i.pop('gt_is_crowd'), tf.bool)
            is_difficult = tf.cast(gt_box_dict_i.pop('gt_difficult'), tf.bool)
            is_valid_bool = is_valid & ~is_crowd & ~is_difficult
            gt_boxlist_i = box_list.BoxList.from_tensor_dict(gt_box_dict_i)
            valid_gt_boxlist_i = box_list_ops.boolean_mask(gt_boxlist_i, is_valid_bool)
            crowd_gt_boxlist_i = box_list_ops.boolean_mask(gt_boxlist_i, is_crowd)
            difficult_gt_boxlist_i = box_list_ops.boolean_mask(gt_boxlist_i, is_difficult)

            anchor_match_matrix = []
            grid_inds = []
            for cell_anchors, stride in zip(cell_anchor_list, strides):
                y_min, x_min, y_max, x_max = tf.split(
                    value=valid_gt_boxlist_i.boxes, num_or_size_splits=4, axis=1)
                y_min = y_min * 1. / stride
                y_max = y_max * 1. / stride
                x_min = x_min * 1. / stride
                x_max = x_max * 1. / stride

                center_y = (y_min + y_max) / 2.
                center_x = (x_min + x_max) / 2.
                height = y_max - y_min
                width = x_max - x_min

                grid_ind_y = tf.cast(tf.floor(center_y), tf.int64)
                grid_ind_x = tf.cast(tf.floor(center_x), tf.int64)
                grid_inds.append(tf.concat([grid_ind_y, grid_ind_x], axis=1))
                zeroed_gt_boxes = tf.concat(
                    [tf.zeros_like(height), tf.zeros_like(width), height, width], axis=1
                )
                zeroed_gt_boxlist_i = box_list.BoxList(zeroed_gt_boxes)

                cell_anchor_boxlist = box_list.BoxList(cell_anchors)
                anchor_match_matrix.append(
                    box_list_ops.pairwise_iou(zeroed_gt_boxlist_i, cell_anchor_boxlist)
                )

            anchor_match_matrix = tf.concat(anchor_match_matrix, axis=1)
            grid_inds = tf.stack(grid_inds)

            predicted_boxlist = box_list.BoxList(predicted_boxes)
            predicted_match_matrix = box_list_ops.pairwise_iou(
                valid_gt_boxlist_i, predicted_boxlist, iou_type="ciou"
            )
            crowd_matrix = box_list_ops.pairwise_iou(
                crowd_gt_boxlist_i, predicted_boxlist, iou_type="ciou"
            )
            difficult_matrix = box_list_ops.pairwise_iou(
                difficult_gt_boxlist_i, predicted_boxlist, iou_type="ciou"
            )
            best_anchor_inds, respond_bgd = self.matcher(
                anchor_match_matrix, predicted_match_matrix, crowd_matrix, difficult_matrix
            )

            level_assigments = tf.floordiv(best_anchor_inds, num_cell_anchors)
            grid_idxs = tf.gather_nd(
                grid_inds,
                tf.stack(
                    [
                        tf.cast(level_assigments, tf.int32),
                        tf.range(tf.shape(level_assigments)[0])
                    ],
                    axis=1
                )
            )
            level_anchor_inds = tf.floormod(best_anchor_inds, num_cell_anchors)

            matched_anchor_inds = tf.concat(
                [grid_idxs, tf.expand_dims(level_anchor_inds, axis=1)], axis=1
            )
            gt_classes = valid_gt_boxlist_i.get_field("gt_classes")
            matched_classes_inds = tf.concat(
                [matched_anchor_inds, tf.expand_dims(gt_classes, axis=1)], axis=1
            )

            respond_bbox = []
            label_prob = []
            target_boxes = []
            for level, grid_size in enumerate(self.grid_sizes):
                level_inds = tf.where(tf.equal(level_assigments, level))[:, 0]
                level_matched_anchor_inds = tf.gather(matched_anchor_inds, level_inds)
                level_sparse_conf_labels = tf.sparse.SparseTensor(
                    indices=level_matched_anchor_inds,
                    values=tf.ones([tf.shape(level_inds)[0]], dtype=tf.float32),
                    dense_shape=[grid_size[0], grid_size[1], 3]
                )
                level_conf_labels = tf.sparse.to_dense(
                    level_sparse_conf_labels, validate_indices=False
                )
                respond_bbox.append(tf.reshape(level_conf_labels, [-1]))

                level_matched_classes_inds = tf.gather(matched_classes_inds, level_inds)
                level_sparse_class_labels = tf.sparse.SparseTensor(
                    indices=level_matched_classes_inds,
                    values=tf.ones([tf.shape(level_inds)[0]], dtype=tf.float32),
                    dense_shape=[grid_size[0], grid_size[1], 3, self.num_classes]
                )
                level_class_labels = tf.sparse.to_dense(
                    level_sparse_class_labels, validate_indices=False
                )
                label_prob.append(tf.reshape(level_class_labels, [-1, self.num_classes]))

                level_gt_boxes = tf.gather(valid_gt_boxlist_i.boxes, level_inds)
                level_boxes = []
                for i in range(4):
                    level_sparse_box_dims = tf.sparse.SparseTensor(
                        indices=level_matched_anchor_inds,
                        values=level_gt_boxes[:, i],
                        dense_shape=[grid_size[0], grid_size[1], 3]
                    )
                    level_box_dims = tf.sparse.to_dense(
                        level_sparse_box_dims, validate_indices=False
                    )
                    level_boxes.append(tf.reshape(level_box_dims, [-1]))
                target_boxes.append(tf.stack(level_boxes, axis=-1))

            respond_bbox = tf.concat(respond_bbox, axis=0)
            label_prob = tf.concat(label_prob, axis=0)
            respond_bgd = (1. - respond_bbox) * tf.cast(respond_bgd, tf.float32)
            target_boxes = tf.concat(target_boxes, axis=0)

            return respond_bgd, respond_bbox, label_prob, target_boxes

        gt_box_dict = self.gt_boxes.as_tensor_dict(trackings=['image_shape'])
        return tf.map_fn(
            get_ground_truth_single_image,
            [gt_box_dict, predicted_boxes],
            dtype=(tf.float32, tf.float32, tf.float32, tf.float32),
            back_prop=False
        )

    def _get_predictions(self):
        boxes_all = []
        probs_all = []
        conf_all = []
        raw_conf_all = []
        raw_prob_all = []
        anchor_list = self.anchor_generator(self.pred_logits)
        for level in range(len(self.pred_logits)):
            logits = self.pred_logits[level]
            anchors = anchor_list[level].boxes
            cell_anchors = self.anchor_generator.cell_anchors[level]
            num_anchors = self.anchor_generator.num_cell_anchors[level]
            stride = self.anchor_generator.strides[level]
            scale_yx = self.scale_yx[level]

            batch_size, height, width, _ = combined_static_and_dynamic_shape(logits)
            logits = tf.reshape(
                logits, [batch_size, height, width, num_anchors, 5 + self.num_classes]
            )
            raw_dydx, raw_dhdw, raw_conf, raw_prob = tf.split(
                logits, [2, 2, 1, self.num_classes], axis=-1
            )

            anchor_grid = tf.tile(
                tf.expand_dims(anchors[:, :2], axis=0), [batch_size, 1, 1]
            )
            anchor_grid = tf.reshape(anchor_grid, [batch_size, height, width, 3, 2])

            dydx = scale_yx * tf.nn.sigmoid(raw_dydx) - 0.5 * (scale_yx - 1)
            pred_yx = (anchor_grid + dydx) * stride
            pred_hw = tf.exp(raw_dhdw) * cell_anchors[:, 2:]
            box_min = pred_yx - 0.5 * pred_hw
            box_max = pred_yx + 0.5 * pred_hw
            pred_boxes = tf.concat([box_min, box_max], axis=-1)

            pred_conf = tf.nn.sigmoid(raw_conf)
            pred_prob = tf.nn.sigmoid(raw_prob)

            pred_prob = pred_conf * pred_prob
            pred_prob = tf.reshape(pred_prob, [batch_size, -1, self.num_classes])
            pred_boxes = tf.reshape(pred_boxes, [batch_size, -1, 4])
            pred_conf = tf.reshape(pred_conf, [batch_size, -1])

            boxes_all.append(pred_boxes)
            conf_all.append(pred_conf)
            probs_all.append(pred_prob)

            raw_conf_all.append(tf.reshape(raw_conf, [batch_size, -1]))
            raw_prob_all.append(tf.reshape(raw_prob, [batch_size, -1, self.num_classes]))

        boxes_all = tf.concat(boxes_all, axis=1)
        conf_all = tf.concat(conf_all, axis=1)
        probs_all = tf.concat(probs_all, axis=1)
        raw_conf_all = tf.concat(raw_conf_all, axis=1)
        raw_prob_all = tf.concat(raw_prob_all, axis=1)

        return boxes_all, conf_all, probs_all, raw_conf_all, raw_prob_all

    def losses(self):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.
        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        pred_boxes, pred_confs, _, pred_conf_logits, pred_cls_logits = self._get_predictions()
        respond_bgd, respond_bbox, label_prob, gt_boxes = self._get_ground_truth(pred_boxes)

        pos_masks = respond_bbox > 0
        neg_masks = respond_bgd > 0
        valid_masks = tf.logical_or(pos_masks, neg_masks)
        num_pos_anchors = tf.count_nonzero(respond_bbox, dtype=tf.int32)
        num_neg_anchors = tf.count_nonzero(respond_bgd, dtype=tf.int32)
        tf.summary.scalar("yolov4/num_pos_anchors", num_pos_anchors / self.num_images)
        tf.summary.scalar("yolov4/num_neg_anchors", num_neg_anchors / self.num_images)

        num_images = tf.cast(self.num_images, tf.float32)
        cls_loss = tf.cond(
            tf.reduce_any(pos_masks),
            lambda: self.cls_normalizer / num_images * tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.boolean_mask(pred_cls_logits, pos_masks),
                    labels=tf.boolean_mask(label_prob, pos_masks)
                )
            ),
            lambda: 0.0
        )

        gt_boxes = tf.boolean_mask(gt_boxes, pos_masks)
        pred_boxes = tf.boolean_mask(pred_boxes, pos_masks)
        gt_boxlist = box_list.BoxList(gt_boxes)
        gt_box_area = box_list_ops.area(gt_boxlist)
        image_area = tf.cast(self.image_shape[0] * self.image_shape[1], tf.float32)
        box_loss_scales = 2.0 - gt_box_area / image_area
        box_loss = tf.cond(
            tf.reduce_any(pos_masks),
            lambda: iou_loss(
                targets=gt_boxes,
                predictions=pred_boxes,
                iou_type='ciou',
                weight=box_loss_scales * self.iou_normalizer / num_images,
                reduction="sum"
            ),
            lambda: 0.0
        )

        conf_focal = tf.boolean_mask(tf.pow(respond_bbox - pred_confs, 2), valid_masks)
        conf_loss = tf.cond(
            tf.reduce_any(valid_masks),
            lambda: 1. / num_images * tf.reduce_sum(
                conf_focal * tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.boolean_mask(pred_conf_logits, valid_masks),
                    labels=tf.boolean_mask(respond_bbox, valid_masks)
                )
            ),
            lambda: 0.0
        )

        losses = {"conf_loss": conf_loss, "cls_loss": cls_loss, "box_loss": box_loss}

        return losses

    def inference(self):
        predicted_boxes, _, predicted_prob, *_ = self._get_predictions()

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
            boxes, probs = args

            score_max = tf.reduce_max(probs, axis=-1)

            # filter out the proposals with low confidence score
            keep_idxs = tf.where(score_max > self.score_threshold)[:, 0]
            predicted_prob = tf.gather(probs, keep_idxs)
            predicted_boxes = tf.gather(boxes, keep_idxs)

            classes_idxs = tf.argmax(predicted_prob, axis=-1, output_type=tf.int64)
            predicted_scores = tf.reduce_max(predicted_prob, axis=-1)

            keep = tf.image.non_max_suppression(
                predicted_boxes, predicted_scores, self.post_nms_topk, self.nms_threshold
            )
            boxes = tf.gather(predicted_boxes, keep, name="boxes")
            scores = tf.gather(predicted_scores, keep, name="scores")
            class_idxs = tf.gather(classes_idxs, keep, name="classes")

            result_boxlist = box_list.BoxList(boxes)
            result_boxlist.add_field('scores', scores)
            result_boxlist.add_field('pred_classes', class_idxs)
            result_boxlist.add_field('is_valid', tf.ones_like(scores, dtype=tf.bool))
            # pad to topk_per_image in case num nmsed boxes < topk_per_image
            result_boxlist = box_list_ops.pad_or_clip_boxlist(result_boxlist, self.post_nms_topk)

            return result_boxlist.as_tensor_dict()

        result_boxlist_dict = tf.map_fn(
            inference_single_image,
            [predicted_boxes, predicted_prob],
            dtype={
                'boxes': tf.float32,
                'scores': tf.float32,
                'pred_classes': tf.int64,
                'is_valid': tf.bool,
            }
        )

        result_boxlist = box_list.BoxList.from_tensor_dict(result_boxlist_dict)
        return result_boxlist
