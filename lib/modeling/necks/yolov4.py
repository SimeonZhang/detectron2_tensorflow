import math
import tensorflow as tf

from ...layers import (
    Layer,
    Conv2D,
    MaxPool2D,
    Upsample,
    ShapeSpec,
    get_norm,
    get_activation
)

from .neck import Neck
from .fpn import assert_strides_are_log2_contiguous
from .build import NECK_REGISTRY
from ...utils.arg_scope import arg_scope, add_arg_scope

slim = tf.contrib.slim


@NECK_REGISTRY.register()
class YOLOV4(Neck):
    """
    This module implements YOLO V4 neck consisting of SPP and PAN.
    """
    def __init__(self, cfg, input_shape, **kwargs):
        super().__init__(**kwargs)
        self.in_features = cfg.MODEL.NECK.IN_FEATURES
        assert len(self.in_features) == 3, self.in_features

        self.in_strides = [input_shape[f].stride for f in self.in_features]
        self.in_channels = [input_shape[f].channels for f in self.in_features]
        assert_strides_are_log2_contiguous(self.in_strides)

        self.out_channels = cfg.MODEL.NECK.OUT_CHANNELS
        self.norm = cfg.MODEL.NECK.NORM
        self.activation = cfg.MODEL.NECK.ACTIVATION

        self.build()

    def build(self):

        use_bias = self.norm == ""
        normalizer = get_norm(self.norm)
        activation = get_activation(self.activation, alpha=0.1)
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            with arg_scope(
                [Conv2D],
                use_bias=use_bias,
                normalizer=normalizer,
                normalizer_params={"scope": "norm"},
                activation=activation,
                weights_initializer=tf.variance_scaling_initializer(scale=1.0)
            ):
                self.process1 = SPP(
                    in_channels=self.in_channels[2],
                    out_channels=self.out_channels * 4,
                    scope="process1"
                )
                self.process2 = TopDown(
                    in_channels=self.in_channels[1],
                    out_channels=self.out_channels * 2,
                    scope="process2"
                )
                self.process3 = TopDown(
                    in_channels=self.in_channels[0],
                    out_channels=self.out_channels,
                    scope="process3"
                )
                self.process4 = BottomUp(
                    out_channels=self.out_channels * 2,
                    scope="process4"
                )
                self.process5 = BottomUp(
                    out_channels=self.out_channels * 4,
                    scope="process5"
                )

        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {
            "p{}".format(int(math.log2(s))): s for s in self.in_strides
        }

        self._out_features = list(sorted(self._out_feature_strides.keys()))
        self._out_feature_channels = {}
        for i, f in enumerate(self._out_features):
            self._out_feature_channels[f] = 2 ** i * self.out_channels
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
        c3, c4, c5 = [bottom_up_features[f] for f in self.in_features]
        l5 = self.process1(c5)
        l4 = self.process2(l5, c4)
        l3 = self.process3(l4, c3)
        l4 = self.process4(l3, l4)
        l5 = self.process5(l4, l5)
        results = [l3, l4, l5]
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@add_arg_scope
class SPP(Layer):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            self.conv1 = Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                scope="conv1")
            self.conv2 = Conv2D(
                in_channels=out_channels,
                out_channels=out_channels * 2,
                kernel_size=3,
                scope="conv2")
            self.conv3 = Conv2D(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=1,
                scope="conv3")

            self.maxpool1 = MaxPool2D(kernel_size=13, scope="pool1")
            self.maxpool2 = MaxPool2D(kernel_size=9, scope="pool1")
            self.maxpool3 = MaxPool2D(kernel_size=5, scope="pool1")

            self.conv4 = Conv2D(
                in_channels=out_channels * 4,
                out_channels=out_channels,
                kernel_size=1,
                scope="conv4")
            self.conv5 = Conv2D(
                in_channels=out_channels,
                out_channels=out_channels * 2,
                kernel_size=3,
                scope="conv5")
            self.conv6 = Conv2D(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=1,
                scope="conv6")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        maxpool1 = self.maxpool1(x)
        maxpool2 = self.maxpool2(x)
        maxpool3 = self.maxpool3(x)
        x = tf.concat([maxpool1, maxpool2, maxpool3, x], axis=3)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


@add_arg_scope
class TopDown(Layer):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            self.conv1 = Conv2D(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=1,
                scope="conv1")
            self.upsample = Upsample(2)

            self.conv2 = Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                scope="conv2")
            self.conv3 = Conv2D(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=1,
                scope="conv3")
            self.conv4 = Conv2D(
                in_channels=out_channels,
                out_channels=out_channels * 2,
                kernel_size=3,
                scope="conv4")
            self.conv5 = Conv2D(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=1,
                scope="conv5")
            self.conv6 = Conv2D(
                in_channels=out_channels,
                out_channels=out_channels * 2,
                kernel_size=3,
                scope="conv6")
            self.conv7 = Conv2D(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=1,
                scope="conv7")

    def call(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.upsample(x1)

        x2 = self.conv2(x2)
        x = tf.concat([x2, x1], axis=3)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x


@add_arg_scope
class BottomUp(Layer):

    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            self.conv1 = Conv2D(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                scope="conv1")

            self.conv2 = Conv2D(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=1,
                scope="conv2")
            self.conv3 = Conv2D(
                in_channels=out_channels,
                out_channels=out_channels * 2,
                kernel_size=3,
                scope="conv3")
            self.conv4 = Conv2D(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=1,
                scope="conv4")
            self.conv5 = Conv2D(
                in_channels=out_channels,
                out_channels=out_channels * 2,
                kernel_size=3,
                scope="conv5")
            self.conv6 = Conv2D(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=1,
                scope="conv6")

    def call(self, x1, x2):
        x1 = self.conv1(x1)
        x = tf.concat([x1, x2], axis=3)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

