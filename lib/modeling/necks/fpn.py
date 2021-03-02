import math
import tensorflow as tf

from ...layers import (
    Layer,
    Conv2D,
    ShapeSpec,
    upsample,
    get_norm
)

from .neck import Neck
from .build import NECK_REGISTRY
from ...utils.arg_scope import arg_scope

slim = tf.contrib.slim


def assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride,
    i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == strides[i - 1] * 2, "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


@NECK_REGISTRY.register()
class FPN(Neck):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, cfg, input_shape, **kwargs):
        super().__init__(**kwargs)
        self.in_features = cfg.MODEL.NECK.IN_FEATURES

        self.in_strides = [input_shape[f].stride for f in self.in_features]
        self.in_channels = [input_shape[f].channels for f in self.in_features]
        assert_strides_are_log2_contiguous(self.in_strides)

        self.out_channels = cfg.MODEL.NECK.OUT_CHANNELS
        self.norm = cfg.MODEL.NECK.NORM
        self.top_block_type = cfg.MODEL.NECK.TOP_BLOCK_TYPE
        self.fuse_type = cfg.MODEL.NECK.FUSE_TYPE
        assert self.fuse_type in {"avg", "sum"}, fuse_type

        self.build()

    def build(self):
        lateral_convs = []
        output_convs = []

        use_bias = self.norm == ""
        normalizer = get_norm(self.norm)
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            with arg_scope([Conv2D],
                           use_bias=use_bias,
                           normalizer=normalizer,
                           normalizer_params={"scope": "norm"},
                           weights_initializer=(
                               tf.variance_scaling_initializer(scale=1.0))
                           ):
                for idx, in_channels in enumerate(self.in_channels):
                    stage = int(math.log2(self.in_strides[idx]))
                    lateral_conv = Conv2D(
                        in_channels=in_channels,
                        out_channels=self.out_channels,
                        kernel_size=1,
                        use_bias=use_bias,
                        normalizer=normalizer,
                        scope="fpn_lateral{}".format(stage))
                    output_conv = Conv2D(
                        in_channels=self.out_channels,
                        out_channels=self.out_channels,
                        kernel_size=3,
                        stride=1,
                        normalizer=normalizer,
                        scope="fpn_output{}".format(stage))

                    lateral_convs.append(lateral_conv)
                    output_convs.append(output_conv)
                # Place convs into top-down order (from low to high resolution)
                # to make the top-down computation in forward clearer.
                self.lateral_convs = lateral_convs[::-1]
                self.output_convs = output_convs[::-1]

            # top block output feature maps.
            if self.top_block_type == "MAXPOOL":
                self.top_block = LastLevelMaxPool(scope="top_block")
            elif self.top_block_type == "P6P7":
                in_channels = self.in_channels[
                    self.in_features.index(LastLevelP6P7.in_feature)
                ]
                self.top_block = LastLevelP6P7(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    scope="top_block"
                )
            else:
                self.top_block = None

        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {
            "p{}".format(int(math.log2(s))): s for s in self.in_strides}

        for s in range(stage, stage + self.top_block.num_levels):
            self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(sorted(self._out_feature_strides.keys()))
        self._out_feature_channels = {k: self.out_channels for k in self._out_features}
        self._size_divisibility = self.in_strides[-1]

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def call(self, bottom_up_features):
        """
        Args:
            bottom_up_features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are height, width, and channels, which may
                vary between feature maps (e.g., if a feature pyramid is used).

        Returns:
            dict[str: Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names
                follow the FPN paper convention: "p<stage>", where stage
                has stride = 2 ** stage e.g., ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order
        # from low to high resolution
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
                x[1:], self.lateral_convs[1:], self.output_convs[1:]):
            top_down_features = upsample(prev_features, factor=2)
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self.fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(
                self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[
                    self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class LastLevelMaxPool(Layer):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """
    num_levels = 1
    in_feature = "p5"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return [slim.max_pool2d(x, kernel_size=1, stride=2, padding="VALID")]


class LastLevelP6P7(Layer):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """
    num_levels = 2
    in_feature = "res5"

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            with arg_scope(
                [Conv2D],
                kernel_size=3,
                stride=2,
                padding="SAME",
                weights_initializer=(tf.variance_scaling_initializer(scale=1.0))
            ):
                self.p6 = Conv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    activation=tf.nn.relu,
                    scope="p6")
                self.p7 = Conv2D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    scope="p7")

    def call(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(p6)
        return [p6, p7]
