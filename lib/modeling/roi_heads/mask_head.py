import tensorflow as tf

from ...layers import Layer, Conv2D, ConvTranspose2D, ShapeSpec, get_norm
from ...utils import shape_utils, tf_utils
from ...utils.registry import Registry
from ...structures import box_list, box_list_ops
from ...utils.arg_scope import arg_scope

ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.
The registered object will be called with `obj(cfg, input_shape)`.
"""


def mask_rcnn_loss(pred_mask_logits, instances, use_mini_masks):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, Hmask, Wmask, C) or (B, Hmask, Wmask, 1)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (SparseBoxList): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    pred_mask_logits = tf.transpose(pred_mask_logits, [0, 3, 1, 2])
    pred_mask_shape = shape_utils.combined_static_and_dynamic_shape(pred_mask_logits)
    total_num_masks = pred_mask_shape[0]

    boxes = instances.data.boxes
    if use_mini_masks:
        # Transform ROI coordinates from absolute image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(instances.data.boxes, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(instances.data.get_field('gt_boxes'), 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)

    gt_masks = tf.expand_dims(instances.data.get_field('gt_masks'), axis=3)
    gt_masks = tf.image.crop_and_resize(
        gt_masks, boxes, tf.range(tf.shape(boxes)[0]), pred_mask_shape[2:]
    )
    gt_masks = tf.round(tf.squeeze(gt_masks, axis=3))

    mask_inds = tf.range(total_num_masks)
    if pred_mask_shape[1] == 1:
        indices = tf.stack([mask_inds, tf.zeros_like(mask_inds)], axis=1)
    else:
        indices = tf.stack([mask_inds, tf.cast(instances.data.get_field('gt_classes'), tf.int32)], axis=1)
    pred_mask_logits = tf.gather_nd(pred_mask_logits, indices)

    mask_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=gt_masks, logits=pred_mask_logits
        )
    )
    return tf.cond(total_num_masks > 0, lambda: mask_loss, lambda: 0.)


def mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, Hmask, Wmask, C) or (B, Hmask, Wmask, 1)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (SparseBoxList): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".
    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    pred_mask_logits = tf.transpose(pred_mask_logits, [0, 3, 1, 2])
    pred_mask_shape = shape_utils.combined_static_and_dynamic_shape(pred_mask_logits)
    cls_agnostic_mask = pred_mask_shape[1] == 1
    total_num_masks = pred_mask_shape[0]

    mask_inds = tf.cast(tf.range(total_num_masks), tf.int64)
    if cls_agnostic_mask:
        indices = tf.stack([mask_inds, tf.ones_like(mask_inds)], axis=1)
    else:
        indices = tf.stack([mask_inds, pred_instances.data.get_field('pred_classes')], axis=1)
    pred_mask_logits = tf.gather_nd(pred_mask_logits, indices)
    mask_probs_pred = tf.nn.sigmoid(pred_mask_logits)
    pred_instances.data.add_field('pred_masks', mask_probs_pred)


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(Layer):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec, **kwargs):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__(**kwargs)

        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK

        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            normalizer = get_norm(norm)
            self.convs = []
            with arg_scope(
                [Conv2D, ConvTranspose2D],
                weights_initializer=tf.variance_scaling_initializer(
                    scale=2.0, mode='fan_out',
                    distribution='untruncated_normal' if tf_utils.get_tf_version_tuple() >= (1, 12) else 'normal'
                ),
                activation=tf.nn.relu,
            ):
                for k in range(num_conv):
                    self.convs.append(
                        Conv2D(
                            input_channels if k == 0 else conv_dims,
                            conv_dims,
                            kernel_size=3,
                            use_bias=not normalizer,
                            normalizer=normalizer,
                            normalizer_params={"channels": conv_dims, "scope": "norm"},
                            padding="SAME",
                            scope="mask_fcn{}".format(k + 1)
                        )
                    )

                self.deconv = ConvTranspose2D(
                    conv_dims if num_conv > 0 else input_channels,
                    conv_dims,
                    kernel_size=2,
                    stride=2,
                    scope="deconv"
                )

                num_mask_classes = 1 if cls_agnostic_mask else num_classes
                self.predictor = Conv2D(
                    conv_dims,
                    num_mask_classes,
                    kernel_size=1,
                    activation=None,
                    weights_initializer=tf.random_normal_initializer(stddev=0.001),
                    scope="predictor"
                )

    def call(self, x):
        for layer in self.convs:
            x = layer(x)
        deconv_features = self.deconv(x)
        return deconv_features, self.predictor(deconv_features)


def build_mask_head(cfg, input_shape, **kwargs):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape, **kwargs)
