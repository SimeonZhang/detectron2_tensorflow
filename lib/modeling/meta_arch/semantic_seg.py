import numpy as np
import tensorflow as tf

from ...data import fields
from ...layers import Layer, Conv2D, GroupNorm, Upsample, ShapeSpec, Sequential, resize_images
from ...utils.registry import Registry
from ...structures.image_list import ImageList
from ...modeling.meta_arch.build import META_ARCH_REGISTRY
from ...utils import shape_utils
from ...utils.arg_scope import arg_scope
from ..backbone import build_backbone
from ..necks import build_neck
from ..postprocessing import sem_seg_postprocess

slim = tf.contrib.slim

SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")
"""
Registry for semantic segmentation heads, which make semantic segmentation predictions
from feature maps.
"""


@META_ARCH_REGISTRY.register()
class SemanticSegmentor(Layer):
    """
    Main class for semantic segmentation architectures.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)

        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg, self.backbone.output_shape(), scope="neck")
        self.sem_seg_head = build_sem_seg_head(
            cfg, self.neck.output_shape(), scope="sem_seg_head"
        )

        pixel_mean = tf.convert_to_tensor(cfg.MODEL.PIXEL_MEAN, dtype=tf.float32)
        pixel_std = tf.convert_to_tensor(cfg.MODEL.PIXEL_STD, dtype=tf.float32)
        self.input_format = cfg.MODEL.INPUT_FORMAT
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.segmentation_output_format = cfg.MODEL.SEGMENTATION_OUTPUT.FORMAT
        self.segmentation_output_resolution = cfg.MODEL.SEGMENTATION_OUTPUT.FIXED_RESOLUTION

    def call(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            sem_seg: semantic segmentation ground truth
            Other information that's included in the original dicts, with tf.vas:
                "height", "width" (int): the output resolution of th with tf.va, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            list[dict]: Each dict is the output for one input image. with tf.va
                The dict contains one key "sem_seg" whose value is a with tf.va
                Tensor of the output resolution that represents the
                per-pixel segmentation prediction.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.neck(self.backbone(images.tensor))

        if "sem_seg" in batched_inputs:
            targets = ImageList.from_tensors(
                batched_inputs['sem_seg'],
                image_shapes,
                self.backbone.size_divisibility,
                self.sem_seg_head.ignore_value
            ).tensor
        else:
            targets = None
        result, losses = self.sem_seg_head(features, targets)

        if self.training:
            return losses

        if self.segmentation_output_format == "fixed":
            output_shape = [
                self.segmentation_output_resolution, self.segmentation_output_resolution
            ]
        else:
            output_shape = tf.shape(images.tensor)
        result = sem_seg_postprocess(
            result, image_shapes, output_shape, self.segmentation_output_format
        )
        result_fields = fields.ResultFields
        return {result_fields.sem_seg: tf.argmax(result, axis=-1)}

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = batched_inputs["image"]
        images = self.normalizer(images)
        if self.input_format == "BGR": images = images[..., ::-1]
        image_shapes = batched_inputs["image_shape"]
        images = ImageList.from_tensors(
            images, image_shapes, self.neck.size_divisibility
        )
        return images


def build_sem_seg_head(cfg, input_shape, **kwargs):
    """
    Build a semantic segmentation head from `cfg.MODEL.SEM_SEG_HEAD.NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape, **kwargs)


@SEM_SEG_HEADS_REGISTRY.register()
class SemSegFPNHead(Layer):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape, **kwargs):
        super().__init__(**kwargs)

        self.in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        feature_strides = {k: v.stride for k, v in input_shape.items()}
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        norm = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT

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
                [Conv2D],
                out_channels=conv_dims,
                kernel_size=3,
                stride=1,
                padding='SAME',
                use_bias=not normalizer,
                normalizer=normalizer,
                normalizer_params=normalizer_params,
                activation=tf.nn.relu,
                weights_initializer=tf.variance_scaling_initializer(),
            ):
                for in_feature in self.in_features:
                    head_ops = Sequential()
                    head_length = max(
                        1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
                    )
                    for k in range(head_length):
                        conv = Conv2D(
                            feature_channels[in_feature] if k == 0 else conv_dims,
                            scope="{}_{}".format(in_feature, 2 * k)
                        )
                        head_ops.add(conv)
                        if feature_strides[in_feature] != self.common_stride:
                            head_ops.add(
                                Upsample(factor=2, method="bilinear")
                            )
                    self.scale_heads.append(head_ops)
                self.predictor = Conv2D(
                    in_channels=conv_dims,
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding='VALID',
                    use_bias=True,
                    normalizer=None,
                    normalizer_params=None,
                    activation=None,
                    weights_initializer=tf.variance_scaling_initializer(),
                    scope="predictor"
                )

    def call(self, features, targets=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        x = self.predictor(x)
        orig_shape = tf.shape(x)[1:3]
        output_shape = [
            orig_shape[0] * self.common_stride, orig_shape[1] * self.common_stride
        ]
        x = resize_images(x, output_shape, align_corners=True)

        if self.training:
            losses = {}
            labels = tf.cast(tf.reshape(targets, shape=[-1]), tf.int64)
            logits = tf.reshape(x, shape=[-1, self.num_classes])
            not_ignore_mask = tf.not_equal(labels, self.ignore_value)
            losses["loss_sem_seg"] = tf.cond(
                tf.reduce_any(not_ignore_mask),
                lambda: tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.boolean_mask(labels, not_ignore_mask),
                        logits=tf.boolean_mask(logits, not_ignore_mask)
                    ) * self.loss_weight
                ),
                lambda: 0.,
                name="loss_sem_seg"
            )
            return x, losses
        else:
            return x, {}
