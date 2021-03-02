from abc import ABCMeta, abstractmethod
import tensorflow as tf

from ...layers import (
    Layer,
    Conv2D,
    DeformConv2D,
    ModulatedDeformConv2D,
    DropBlock,
    drop_connect
)
from ...utils.arg_scope import add_arg_scope

slim = tf.contrib.slim


class BaseBlock(Layer, metaclass=ABCMeta):

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        stride=1,
        rate=1,
        num_groups=1,
        stride_in_1x1=False,
        drop_connect_rate=None,
        activation=tf.nn.relu,
        outputs_collections=None,
        **kwargs
    ):
        super(BaseBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bottleneck_channels=bottleneck_channels,
            stride=stride,
            rate=rate,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            drop_connect_rate=drop_connect_rate,
            activation=activation,
            outputs_collections=outputs_collections,
            **kwargs
        )
        self.build()

    @abstractmethod
    def build(self):
        pass

    def call(self, inputs):
        if self.shortcut is None:
            shortcut = inputs
        else:
            shortcut = self.shortcut(inputs)
        shortcut = self.dropblock(shortcut)

        residual = self.conv1(inputs)
        residual = self.dropblock(residual)
        residual = self.conv2(residual)
        residual = self.dropblock(residual)
        residual = self.conv3(residual)
        residual = self.dropblock(residual)

        if self.drop_connect_rate:
            residual = drop_connect(residual, self.training, self.drop_connect_rate)

        output = shortcut + residual
        if self.activation is not None:
            output = self.activation(output)

        return slim.utils.collect_named_outputs(self.outputs_collections, self.sc.name, output)


@add_arg_scope
class ResidualBlock(BaseBlock):

    def __init__(self, drop_connect_rate=None, **kwargs):
        super(ResidualBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            rate=rate,
            num_groups=num_groups,
            outputs_collections=outputs_collections,
            **kwargs
        )
        self.build()

    def build(self):
        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:

            self.shortcut = None
            if self.in_channels != self.out_channels:
                self.shortcut = Conv2D(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=self.stride,
                    activation=None,
                    scope="shortcut")

            self.conv1 = Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=self.stride,
                num_groups=self.num_groups,
                rate=self.rate,
                scope="conv1")

            self.conv2 = Conv2D(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                activation=None,
                scope="conv2")

    def call(self, inputs):

        if self.shortcut is None:
            shortcut = inputs
        else:
            shortcut = self.shortcut(inputs)
        shortcut = self.dropblock(shortcut)

        residual = self.conv1(inputs)
        residual = self.dropblock(residual)
        residual = self.conv2(residual)
        residual = self.dropblock(residual)

        if self.drop_connect_rate:
            residual = drop_connect(residual, self.training, self.drop_connect_rate)

        if activation:
            output = activation(shortcut + residual)

        return slim.utils.collect_named_outputs(
            self.outputs_collections, self.sc.name, output)


@add_arg_scope
class BottleneckBlock(BaseBlock):

    def build(self):
        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:

            self.shortcut = None
            if self.in_channels != self.out_channels:
                self.shortcut = Conv2D(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=self.stride,
                    activation=None,
                    scope="shortcut")

            stride_1x1, stride_3x3 = 1, self.stride
            if self.stride_in_1x1:
                stride_1x1, stride_3x3 = self.stride, 1

            self.conv1 = Conv2D(
                in_channels=self.in_channels,
                out_channels=self.bottleneck_channels,
                kernel_size=1,
                stride=stride_1x1,
                scope="conv1")

            self.conv2 = Conv2D(
                in_channels=self.bottleneck_channels,
                out_channels=self.bottleneck_channels,
                kernel_size=3,
                stride=stride_3x3,
                num_groups=self.num_groups,
                rate=self.rate,
                scope="conv2")

            self.conv3 = Conv2D(
                in_channels=self.bottleneck_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                activation=None,
                scope="conv3")

            self.dropblock = DropBlock()


@add_arg_scope
class DeformBottleneckBlock(BaseBlock):

    def __init__(self, deform_modulated=False, deform_num_groups=1, **kwargs):
        super(DeformBottleneckBlock, self).__init__(
            deform_modulated=deform_modulated,
            deform_num_groups=deform_num_groups,
            **kwargs
        )

    def build(self):
        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:

            self.shortcut = None
            if self.in_channels != self.out_channels:
                self.shortcut = Conv2D(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=self.stride,
                    activation=None,
                    scope="shortcut")

            stride_1x1, stride_3x3 = 1, self.stride
            if self.stride_in_1x1:
                stride_1x1, stride_3x3 = self.stride, 1

            self.conv1 = Conv2D(
                in_channels=self.in_channels,
                out_channels=self.bottleneck_channels,
                kernel_size=1,
                stride=stride_1x1,
                scope="conv1")

            deform_conv_op = DeformConv2D
            if self.deform_modulated:
                deform_conv_op = ModulatedDeformConv2D
            self.conv2 = deform_conv_op(
                in_channels=self.bottleneck_channels,
                out_channels=self.bottleneck_channels,
                kernel_size=3,
                stride=stride_3x3,
                num_groups=self.num_groups,
                deform_num_groups=self.deform_num_groups,
                rate=self.rate,
                scope="conv2")

            self.conv3 = Conv2D(
                in_channels=self.bottleneck_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                activation=None,
                scope="conv3")

            self.dropblock = DropBlock()

