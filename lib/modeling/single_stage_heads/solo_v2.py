import math
import tensorflow as tf
import numpy as np

from ...layers import (
    Layer,
    Sequential,
    Conv2D,
    DeformConv2D,
    ModulatedDeformConv2D,
    GroupNorm,
    Upsample,
    flatten,
    sigmoid_focal_loss,
    dice_loss,
    matrix_nms,
    resize_images
)
from ...utils.shape_utils import combined_static_and_dynamic_shape, pad_or_clip_tensor
from ...utils.arg_scope import arg_scope
from ...structures import box_list, box_list_ops
from .build import SINGLE_STAGE_HEADS_REGISTRY

slim = tf.contrib.slim

__all__ = ["SOLOv2Head"]


def point_nms(inputs, kernel_size=2, scope=None):
    assert kernel_size == 2
    with tf.name_scope(scope, "PointNMS"):
        pooled = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]])
        pooled = tf.nn.max_pool(
            pooled,
            [1, kernel_size, kernel_size, 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        keep = tf.cast(tf.equal(inputs, pooled[:, :-1, :-1, :]), tf.float32)
        return inputs * keep


def center_of_mass(masks, scope=None):
    """
    Get the center coordinate indices of the masks.
    Args:
        masks: [N, height, width]
    Return:
        coor: [N, 2]
    """
    with tf.name_scope(scope, "CenterOfMass"):
        mask_shape = combined_static_and_dynamic_shape(masks)
        xx, yy = tf.meshgrid(tf.range(mask_shape[2]), tf.range(mask_shape[1]))
        coords = tf.stack([tf.reshape(yy, [-1]), tf.reshape(xx, [-1])], axis=1)
        coords = tf.cast(tf.expand_dims(coords, axis=0), tf.float32)

        mask_shape = combined_static_and_dynamic_shape(masks)
        flatten_masks = tf.reshape(
            masks, [mask_shape[0], mask_shape[1] * mask_shape[2], 1]
        )
        flatten_masks = tf.cast(flatten_masks, tf.float32)

        center_of_mass = tf.reduce_mean(flatten_masks * coords, axis=1)
        return center_of_mass[:, 0], center_of_mass[:, 1]


@SINGLE_STAGE_HEADS_REGISTRY.register()
class SOLOv2Head(Layer):
    """
    Implement SOLO v2 Head (https://arxiv.org/abs/2003.10152).
    """

    def __init__(self, cfg, input_shape, **kwargs):
        super().__init__(**kwargs)

        self.mask_kernel_branch = MaskKernelBranch(cfg, input_shape, scope="mask_kernel")
        self.mask_feature_branch = MaskFeatureBranch(cfg, input_shape, scope="mask_feature")


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
            targets (BoxList, optional): Dense `BoxList` contains the ground-truth per-instance 
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
        image_shape = tf.shape(images.tensor)[1:3]
        del images
        
        pred_classes, pred_kernels = self.mask_kernel_branch(features)
        pred_mask_features = self.mask_feature_branch(features)

        if self.training:
            losses = self.mask_kernel_branch.losses(
                pred_classes, pred_kernels, pred_mask_features, targets
            )
            return None, losses
        else:
            results = self.mask_kernel_branch.inference(
                pred_classes, pred_kernels, pred_mask_features, image_shape
            )
            return results, {}


class MaskKernelBranch(Layer):
    """
    The mask kernel brach, along wth the semantic category branch.
    The mask kernel brach and semantic category branch have a common structure 
    but separate parameters.
    """

    def __init__(self, cfg, input_shape, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = cfg.MODEL.SINGLE_STAGE_HEAD.NUM_CLASSES
        self.in_features = cfg.MODEL.SINGLE_STAGE_HEAD.IN_FEATURES
        self.strides = [input_shape[f].stride for f in self.in_features]
        self.sigma = cfg.MODEL.SOLO.SIGMA
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.SOLO.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.SOLO.FOCAL_LOSS_GAMMA
        self.ins_loss_weight = cfg.MODEL.SOLO.INS_LOSS_WEIGHT
        # Inference parameters:
        self.score_threshold = cfg.MODEL.SOLO.SCORE_THRESH_TEST
        self.update_score_threshold = cfg.MODEL.SOLO.UPDATE_SCORE_THRESH_TEST
        self.pre_nms_topk = cfg.MODEL.SOLO.TOPK_CANDIDATES_TEST
        self.mask_threshold = cfg.MODEL.SOLO.MASK_THRESH_TEST
        self.nms_kernel = cfg.MODEL.SOLO.NMS_KERNEL
        self.nms_sigma = cfg.MODEL.SOLO.NMS_SIGMA
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        self.scale_ranges = cfg.MODEL.SOLO.SCALE_RANGES
        self.num_grids = cfg.MODEL.SOLO.NUM_GRIDS
        self.mask_kernel_size = cfg.MODEL.SOLO.MASK_KERNEL_SIZE
        self.mask_feature_out_dims = cfg.MODEL.SOLO.MASK_FEATURE_OUT_DIMS

        in_channels = input_shape[self.in_features[0]].channels
        num_convs = cfg.MODEL.SOLO.MASK_KERNEL_NUM_CONVS
        conv_dims = cfg.MODEL.SOLO.MASK_KERNEL_CONVS_DIM
        kernel_dims = self.mask_feature_out_dims * self.mask_kernel_size ** 2
        norm = cfg.MODEL.SOLO.MASK_KERNEL_NORM
        prior_prob = cfg.MODEL.SOLO.PRIOR_PROB

        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            cls_subnet = []
            kernel_subnet = []
            if norm == "GN":
                normalizer = GroupNorm
                normalizer_params = {
                    'channels': conv_dims,
                    'num_groups': 32,
                    'scope': 'norm'
                }
            else:
                normalizer = None
                normalizer_params = {}
            with arg_scope(
                [Conv2D, DeformConv2D, ModulatedDeformConv2D],
                kernel_size=3,
                stride=1,
                padding='SAME',
                use_bias=not normalizer,
                normalizer=normalizer,
                normalizer_params=normalizer_params,
                activation=tf.nn.relu,
                weights_initializer=tf.random_normal_initializer(stddev=0.01),
            ):
                conv_op = Conv2D
                if cfg.MODEL.SOLO.USE_DEFORM_CONV:
                    conv_op = DeformConv2D
                    if cfg.MODEL.SOLO.DEFORM_MODULATED:
                        conv_op = ModulatedDeformConv2D
                for i in range(num_convs):
                    cls_in_channels = in_channels if i == 0 else conv_dims
                    kernel_in_channels = in_channels + 2 if i == 0 else conv_dims
                    cls_subnet.append(
                        conv_op(cls_in_channels, conv_dims, scope=f"cate_subnet{2*i}")
                    )
                    kernel_subnet.append(
                        conv_op(kernel_in_channels, conv_dims, scope=f"kernel_subnet{2*i}")
                    )

                self.cls_subnet = Sequential(cls_subnet)
                self.kernel_subnet = Sequential(kernel_subnet)
                # Use prior in model initialization to improve stability
                bias_initializer = tf.constant_initializer(-math.log((1 - prior_prob) / prior_prob))
                self.solo_cate = Conv2D(
                    conv_dims,
                    self.num_classes,
                    activation=None,
                    use_bias=True,
                    normalizer=None,
                    bias_initializer=bias_initializer,
                    scope="solo_cate"
                )
                self.solo_kernel = Conv2D(
                    conv_dims,
                    kernel_dims,
                    activation=None,
                    use_bias=True,
                    normalizer=None,
                    scope="solo_kernel"
                )

    def split_features(self, features):
        new_features = [
            resize_images(
                features[self.in_features[0]],
                tf.shape(features[self.in_features[1]])[1:3],
                align_corners=True
            ),
            features[self.in_features[1]],
            features[self.in_features[2]],
            features[self.in_features[3]],
            resize_images(
                features[self.in_features[4]],
                tf.shape(features[self.in_features[3]])[1:3],
                align_corners=True
            )
        ]
        self.strides[0] *= 2
        self.strides[-1] /= 2
        return new_features

    def call(self, features):
        """
        Arguments:
            features (Dict {str: Teonsor}): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.
        Returns:
            pred_classes (list[Tensor]): #lvl tensors, each has shape (N, Si, Si, K).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            pred_kernels (list[Tensor]): #lvl tensors, each has shape (N, Si, Si, D).
        """
        pred_classes = []
        pred_kernels = []
        new_features = self.split_features(features)
        for i in range(len(self.in_features)):
            shape = combined_static_and_dynamic_shape(new_features[i])
            y = tf.linspace(-1., 1., shape[1])
            x = tf.linspace(-1., 1., shape[2])
            xx, yy = tf.meshgrid(x, y)
            yy = tf.tile(yy[None, ..., None], [shape[0], 1, 1, 1])
            xx = tf.tile(xx[None, ..., None], [shape[0], 1, 1, 1])
            feature = tf.concat([new_features[i], xx, yy], 3)
            feature = resize_images(
                feature, [self.num_grids[i], self.num_grids[i]], align_corners=True
            )
            pred_cls = self.solo_cate(self.cls_subnet(feature[..., :-2]))
            if not self.training:
                pred_cls = point_nms(tf.nn.sigmoid(pred_cls))
            pred_classes.append(pred_cls)
            pred_kernels.append(self.solo_kernel(self.kernel_subnet(feature)))
        return pred_classes, pred_kernels

    def losses(self, pred_classes, pred_kernels, pred_mask_features, targets):
        gt_boxlist = box_list.SparseBoxList.from_dense(targets)
        pred_mask_size = tf.shape(pred_mask_features)[1:3]
        gt_classes_list, gt_masks_list, training_grid_inds_list = (
            self.get_ground_truth(gt_boxlist, pred_mask_size)
        )

        size_trans = np.power(self.num_grids, 2).cumsum()
        # generate masks
        def generate_masks(
            pred_kernels,
            pred_mask_features,
            training_inds_list
            ):
            
            def generate_single_masks_fn(args):
                kernel, mask_features = args
                feature = tf.expand_dims(mask_features, axis=0)
                # kernel_shape = combined_static_and_dynamic_shape(kernel)
                kernel = tf.reshape(
                    tf.transpose(kernel), 
                    [
                        self.mask_kernel_size, self.mask_kernel_size,
                        self.mask_feature_out_dims, size_trans[-1]
                    ]
                )
                pred_mask = tf.nn.conv2d(
                    feature, kernel, strides=[1, 1, 1, 1], padding='VALID'
                )
                pred_mask = tf.squeeze(pred_mask, axis=0)
                pred_mask = tf.transpose(pred_mask, [2, 0, 1])
                return pred_mask
            
            pred_masks = tf.map_fn(
                generate_single_masks_fn,
                [pred_kernels, pred_mask_features],
                dtype=tf.float32
            )
            pred_masks = tf.gather_nd(pred_masks, training_inds_list)
            pred_masks.set_shape([None, None, None])
            return pred_masks

        pred_kernels_list = []
        training_inds_list = []
        for i, (pred_kernel, training_grid_inds, num_grid) in enumerate(
            zip(pred_kernels, training_grid_inds_list, self.num_grids)
        ):
            kernel_shape = combined_static_and_dynamic_shape(pred_kernel)
            pred_kernel = tf.reshape(
                pred_kernel, [kernel_shape[0], num_grid * num_grid, kernel_shape[-1]]
            )
            # pred_kernel = tf.gather_nd(pred_kernel, training_grid_inds)
            pred_kernels_list.append(pred_kernel)
            if i == 0:
                new_grid_idx = training_grid_inds[:, 1]
            else:
                new_grid_idx = size_trans[i-1] + training_grid_inds[:, 1]
            training_inds = tf.stack([training_grid_inds[:, 0], new_grid_idx], axis=1)
            training_inds_list.append(training_inds)

        pred_kernels_list = tf.concat(pred_kernels_list, axis=1)
        training_inds_list = tf.concat(training_inds_list, axis=0)
        pred_masks = generate_masks(
            pred_kernels_list, pred_mask_features, training_inds_list
        )

        # dice loss
        gt_masks = tf.concat(gt_masks_list, axis=0)
        loss_ins = dice_loss(
            predictions=tf.nn.sigmoid(pred_masks), targets=gt_masks, reduction="mean"
        )
        loss_ins = self.ins_loss_weight * loss_ins

        # cate
        flatten_gt_classes_list = [
            tf.reshape(gt_classes, [-1]) for gt_classes in gt_classes_list
        ]
        flatten_gt_classes = tf.concat(flatten_gt_classes_list, axis=0)
        flatten_gt_classes = tf.one_hot(flatten_gt_classes, self.num_classes + 1)[:, 1:]

        flatten_pred_classes_list = [
            tf.reshape(pred_cls, [-1, self.num_classes]) for pred_cls in pred_classes
        ]
        flatten_pred_classes = tf.concat(flatten_pred_classes_list, axis=0)
        loss_cls = sigmoid_focal_loss(
            predictions=flatten_pred_classes,
            targets=flatten_gt_classes,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        num_ins = tf.add_n(
            [tf.shape(training_inds)[0] for training_inds in training_grid_inds_list]
        )
        loss_cls = loss_cls / tf.cast(num_ins + 1, tf.float32)
        
        return {"loss_ins": loss_ins, "loss_cls": loss_cls}
    
    def get_ground_truth(self, gt_boxlist, pred_mask_size):

        gt_area_sqrt = tf.sqrt(box_list_ops.area(gt_boxlist.data))

        gt_height, gt_width = box_list_ops.height_width(gt_boxlist.data)
        half_h = 0.5 * gt_height * self.sigma
        half_w = 0.5 * gt_width * self.sigma
        gt_boxlist.data.add_field("half_h", half_h)
        gt_boxlist.data.add_field("half_w", half_w)

        gt_classes_list = []
        gt_masks_list = []
        training_grid_inds_list = []

        # get gt for each level
        for (level_lower_bound, level_upper_bound), level_num_grid in zip(self.scale_ranges, self.num_grids):

            level_inds = tf.where(
                tf.logical_and(
                    tf.greater_equal(gt_area_sqrt, level_lower_bound),
                    tf.less_equal(gt_area_sqrt, level_upper_bound)
                    )
            )[:, 0]

            level_gt_boxlist = box_list_ops.gather(gt_boxlist.data, level_inds)

            # mass center
            upsampled_size = tf.cast([pred_mask_size[0] * 4, pred_mask_size[1] * 4], tf.float32)
            center_h, center_w = center_of_mass(level_gt_boxlist.get_field("gt_masks"))
            coord_h = tf.round((center_h / upsampled_size[0]) // (1. / level_num_grid))
            coord_w = tf.round((center_w / upsampled_size[1]) // (1. / level_num_grid))

            # left, top, right, down
            half_h = level_gt_boxlist.get_field("half_h")
            half_w = level_gt_boxlist.get_field("half_w")
            top = tf.maximum(
                coord_h - 1,
                tf.maximum(
                    0.,
                    tf.round(((center_h - half_h) / upsampled_size[0]) // (1. / level_num_grid))
                )
            )[..., None, None, None]
            down = tf.minimum(
                coord_h + 1,
                tf.minimum(
                    level_num_grid - 1.,
                    tf.round(((center_h + half_h) / upsampled_size[0]) // (1. / level_num_grid))
                )
            )[..., None, None, None]
            left = tf.maximum(
                coord_w - 1,
                tf.maximum(
                    0.,
                    tf.round(((center_w - half_w) / upsampled_size[1]) // (1. / level_num_grid))
                )
            )[..., None, None, None]
            right = tf.minimum(
                coord_w + 1,
                tf.minimum(
                    level_num_grid - 1.,
                    tf.round(((center_w + half_w) / upsampled_size[1]) // (1. / level_num_grid))
                )
            )[..., None, None, None]

            xx, yy = tf.meshgrid(tf.range(level_num_grid), tf.range(level_num_grid))
            yy = tf.cast(yy[None, ..., None], tf.float32)
            xx = tf.cast(xx[None, ..., None], tf.float32)

            positive_grid = ((yy >= top) & (yy <= down)) & ((xx >= left) & (xx <= right))
            positive_grid_inds = tf.where(positive_grid)

            training_batch_idx = tf.gather(
                gt_boxlist.indices[:, 0], tf.gather(level_inds, positive_grid_inds[:, 0])
            )
            training_grid_idx = positive_grid_inds[:, 1] * level_num_grid + positive_grid_inds[:, 2]
            training_grid_inds = tf.stack([training_batch_idx, training_grid_idx], axis=1)
            training_grid_inds_list.append(training_grid_inds)
            
            # level_gt_classes = level_gt_boxlist.get_field("gt_classes")[..., None, None, None]
            # level_gt_classes = tf.cast(positive_grid, level_gt_classes.dtype) * level_gt_classes
            level_gt_classes = level_gt_boxlist.get_field("gt_classes")
            level_gt_classes = tf.gather(level_gt_classes, positive_grid_inds[:, 0])
            level_gt_classes = tf.sparse.SparseTensor(
                values=level_gt_classes,
                indices=tf.stack(
                    [training_batch_idx, positive_grid_inds[:, 1], positive_grid_inds[:, 2]], axis=1
                ),
                dense_shape=tf.stack([gt_boxlist.dense_shape[0], level_num_grid, level_num_grid])
            )
            gt_classes_list.append(
                tf.sparse.to_dense(tf.sparse.reorder(level_gt_classes), validate_indices=False)
            )

            level_gt_masks = level_gt_boxlist.get_field("gt_masks")
            level_gt_masks = tf.gather(level_gt_masks, positive_grid_inds[:, 0])
            level_gt_masks = resize_images(
                level_gt_masks[..., None], pred_mask_size, align_corners=True
            )
            level_gt_masks = tf.round(tf.squeeze(level_gt_masks, axis=3))
            gt_masks_list.append(level_gt_masks)

        return gt_classes_list, gt_masks_list, training_grid_inds_list

    def inference(self, pred_probs, pred_kernels, pred_mask_features, image_shape):

        def inference_single_image(args):
            pred_scores, pred_kernels, pred_mask_features = args

            # filter by score thresh
            keep_inds =  tf.where(pred_scores > self.score_threshold)
            pred_scores = tf.gather_nd(pred_scores, keep_inds)

            # pred class labels and kernels
            pred_class_labels = keep_inds[:, 1]
            pred_kernels = tf.gather(pred_kernels, keep_inds[:, 0])

            # trans vector
            size_trans = np.power(self.num_grids, 2).cumsum()
            strides = np.ones(size_trans[-1], dtype=np.float32)

            n_stage = len(self.num_grids)
            strides[:size_trans[0]] *= self.strides[0]
            for i in range(1, n_stage):
                strides[size_trans[i - 1]: size_trans[i]] *= self.strides[i]
            strides = tf.gather(strides, keep_inds[:, 0])

            # mask encoding
            pred_mask_features = tf.expand_dims(pred_mask_features, axis=0)
            kernel_shape = combined_static_and_dynamic_shape(pred_kernels)
            pred_kernels = tf.reshape(
                tf.transpose(pred_kernels, [1, 0]),
                [
                    self.mask_kernel_size, self.mask_kernel_size,
                    self.mask_feature_out_dims, kernel_shape[0]
                ]
            )
            pred_mask_logits = tf.nn.conv2d(
                pred_mask_features, pred_kernels, strides=[1, 1, 1, 1], padding='VALID'
            )
            # mask
            pred_mask_scores = tf.nn.sigmoid(
                tf.transpose(tf.squeeze(pred_mask_logits, axis=0), [2, 0, 1])
            )
            pred_masks = tf.cast(pred_mask_scores > self.mask_threshold, tf.float32)
            sum_masks = tf.reduce_sum(pred_masks, axis=[1, 2])

            # filter by sum masks
            keep_bool = sum_masks > strides

            pred_masks = tf.boolean_mask(pred_masks, keep_bool)
            pred_mask_scores = tf.boolean_mask(pred_mask_scores, keep_bool)
            pred_scores = tf.boolean_mask(pred_scores, keep_bool)
            pred_class_labels = tf.boolean_mask(pred_class_labels, keep_bool)
            sum_masks = tf.boolean_mask(sum_masks, keep_bool)

            # mask scoring
            mask_scoring = tf.reduce_sum(
                (pred_mask_scores * pred_masks), axis=[1, 2]
            ) / sum_masks
            pred_scores = pred_scores * mask_scoring

            # Keep top k top scoring indices only.
            pre_nms_topk = tf.minimum(self.pre_nms_topk, tf.shape(pred_scores)[0])
            pred_scores, topk_idxs = tf.nn.top_k(pred_scores, k=pre_nms_topk, sorted=True)
            pred_masks = tf.gather(pred_masks, topk_idxs)
            sum_masks = tf.gather(sum_masks, topk_idxs)
            pred_class_labels = tf.gather(pred_class_labels, topk_idxs)

            # Matrix NMS
            pred_scores = matrix_nms(
                pred_masks, pred_class_labels, pred_scores, sum_masks=sum_masks,
                kernel=self.nms_kernel, sigma=self.nms_sigma
            )

            # filter by update score thresh
            keep_bool = pred_scores > self.update_score_threshold
            pred_masks = tf.boolean_mask(pred_masks, keep_bool)
            pred_classes = tf.boolean_mask(pred_class_labels, keep_bool)
            pred_scores = tf.boolean_mask(pred_scores, keep_bool)
            is_valid = tf.ones_like(pred_classes, dtype=tf.bool)

            pred_masks = pad_or_clip_tensor(pred_masks, self.max_detections_per_image)
            pred_classes = pad_or_clip_tensor(pred_classes, self.max_detections_per_image)
            pred_scores = pad_or_clip_tensor(pred_scores, self.max_detections_per_image)
            is_valid = pad_or_clip_tensor(is_valid, self.max_detections_per_image)

            results = {
                'pred_classes': pred_classes,
                'pred_masks': pred_masks,
                'scores': pred_scores,
                'is_valid': is_valid,
            }
            return results

        pred_prob_list = []
        pred_kernel_list = []
        for pred_prob, pred_kernel in zip(pred_probs, pred_kernels):
            prob_shape = combined_static_and_dynamic_shape(pred_prob)
            pred_prob_list.append(
                tf.reshape(
                    pred_prob,
                    [prob_shape[0], prob_shape[1]*prob_shape[2], prob_shape[3]]
                )
            )

            kernel_shape = combined_static_and_dynamic_shape(pred_kernel)
            pred_kernel_list.append(
                tf.reshape(
                    pred_kernel,
                    [kernel_shape[0], kernel_shape[1]*kernel_shape[2], kernel_shape[3]]
                )
            )
        pred_probs = tf.concat(pred_prob_list, axis=1)
        pred_kernels = tf.concat(pred_kernel_list, axis=1)
        result_dict = tf.map_fn(
            inference_single_image,
            [pred_probs, pred_kernels, pred_mask_features],
            dtype={
                'pred_classes': tf.int64,
                'pred_masks': tf.float32,
                'scores': tf.float32,
                'is_valid': tf.bool,
            }
        )

        pred_masks = tf.transpose(result_dict["pred_masks"], [0, 2, 3, 1])
        pred_masks = resize_images(pred_masks, image_shape, align_corners=True)
        pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])
        pred_masks = tf.cast(pred_masks > self.mask_threshold, tf.float32)
        result_dict["pred_masks"] = pred_masks

        # generate boxes from masks
        xx, yy = tf.meshgrid(tf.range(image_shape[1]), tf.range(image_shape[0]))

        yy = pred_masks * tf.cast(yy[None, None, ...], dtype=tf.float32)
        xx = pred_masks * tf.cast(xx[None, None, ...], dtype=tf.float32)

        sum_masks = tf.reduce_sum(pred_masks, axis=[2, 3], keep_dims=True)
        yy_mean = tf.reduce_sum(yy, axis=[2, 3], keep_dims=True) / (sum_masks + 1e-5)
        xx_mean = tf.reduce_sum(xx, axis=[2, 3], keep_dims=True) / (sum_masks + 1e-5)
        yy_mean = yy_mean * tf.ones_like(pred_masks)
        xx_mean = xx_mean * tf.ones_like(pred_masks)

        yy = tf.where(yy > 0, yy, yy_mean)
        xx = tf.where(xx > 0, xx, xx_mean)

        ymin = tf.reduce_min(yy, axis=[2, 3])
        xmin = tf.reduce_min(xx, axis=[2, 3])
        ymax = tf.reduce_max(yy, axis=[2, 3])
        xmax = tf.reduce_max(xx, axis=[2, 3])
        pred_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=2)

        result_dict["boxes"] = pred_boxes
        result_boxlist = box_list.BoxList.from_tensor_dict(result_dict)
        return result_boxlist


class MaskFeatureBranch(Layer):
    """
    The mask feature branch. It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape, **kwargs):
        super().__init__(**kwargs)

        self.in_features = cfg.MODEL.SOLO.MASK_FEATURE_IN_FEATURES
        feature_strides = {k: v.stride for k, v in input_shape.items()}
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        conv_dims = cfg.MODEL.SOLO.MASK_FEATURE_CONVS_DIM
        out_dims = cfg.MODEL.SOLO.MASK_FEATURE_OUT_DIMS
        self.common_stride = cfg.MODEL.SOLO.MASK_FEATURE_COMMON_STRIDE
        norm = cfg.MODEL.SOLO.MASK_FEATURE_NORM

        self.scale_heads = []
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            if norm == "GN":
                normalizer = GroupNorm
                normalizer_params = {
                    'channels': conv_dims,
                    'num_groups': 32,
                    'scope': 'norm'
                }
            else:
                normalizer = None
                normalizer_params = {}
            with arg_scope(
                [Conv2D, DeformConv2D, ModulatedDeformConv2D],
                kernel_size=3,
                stride=1,
                padding='SAME',
                use_bias=not normalizer,
                normalizer=normalizer,
                normalizer_params=normalizer_params,
                activation=tf.nn.relu,
                weights_initializer=tf.random_normal_initializer(stddev=0.01),
            ):
                conv_op = Conv2D
                if cfg.MODEL.SOLO.USE_DEFORM_CONV:
                    conv_op = DeformConv2D
                    if cfg.MODEL.SOLO.DEFORM_MODULATED:
                        conv_op = ModulatedDeformConv2D
                for in_feature in self.in_features:
                    head_ops = Sequential()
                    head_length = max(
                        1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
                    )
                    in_channels = feature_channels[in_feature]
                    if in_feature == self.in_features[-1]:
                        in_channels += 2
                    for k in range(head_length):
                        conv = conv_op(
                            in_channels=in_channels if k == 0 else conv_dims,
                            out_channels=conv_dims,
                            scope="{}_{}".format(in_feature, 2 * k)
                        )
                        head_ops.add(conv)
                        if feature_strides[in_feature] != self.common_stride:
                            head_ops.add(
                                Upsample(factor=2, method="bilinear")
                            )
                    self.scale_heads.append(head_ops)
                self.predictor = conv_op(
                    in_channels=conv_dims,
                    out_channels=out_dims,
                    kernel_size=1,
                    stride=1,
                    normalizer_params={'channels': out_dims, "scope": "norm"},
                    padding='VALID',
                    scope="predictor"
                )

    def call(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                res = self.scale_heads[i](features[f])
            else:
                if f == self.in_features[-1]:
                    shape = combined_static_and_dynamic_shape(features[f])
                    y = tf.linspace(-1., 1., shape[1])
                    x = tf.linspace(-1., 1., shape[2])
                    xx, yy = tf.meshgrid(x, y)
                    yy = tf.tile(yy[None, ..., None], [shape[0], 1, 1, 1])
                    xx = tf.tile(xx[None, ..., None], [shape[0], 1, 1, 1])
                    feature = tf.concat([features[f], xx, yy], 3)
                else:
                    feature = features[f]
                res = res + self.scale_heads[i](feature)
        return self.predictor(res)
