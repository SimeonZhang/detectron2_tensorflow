from contextlib import ExitStack, contextmanager
import copy
import tensorflow as tf

from ...layers import Layer, Conv2D, BatchNorm, get_norm, get_activation
from .blocks import BaseBlock
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from ...utils.arg_scope import arg_scope, add_arg_scope

slim = tf.contrib.slim


@contextmanager
def darknet_arg_scope(freeze, norm, activation):
    normalizer = get_norm(norm)
    activation = get_activation(activation)
    with arg_scope(
        [Conv2D],
        use_bias=False,
        normalizer=normalizer,
        normalizer_params={"scope": "norm"},
        activation=activation,
        weights_initializer=tf.variance_scaling_initializer(
            scale=2.0, mode='fan_out'
        )
    ), ExitStack() as stack:
        if norm in ['FrozenBN', 'SyncBN']:
            if freeze or norm == 'FrozenBN':
                stack.enter_context(arg_scope([BatchNorm], training=False))
            else:
                stack.enter_context(arg_scope([BatchNorm], sync=True))

        if freeze:
            stack.enter_context(arg_scope([Conv2D, BatchNorm], trainable=False))

        yield


@add_arg_scope
class DarkNetResidualBlock(BaseBlock):

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        outputs_collections=None,
        **kwargs
    ):
        assert in_channels == out_channels, (in_channels, bottleneck_channels, out_channels)
        super(DarkNetResidualBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bottleneck_channels=bottleneck_channels,
            outputs_collections=outputs_collections,
            **kwargs
        )

    def build(self):
        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:
            self.conv1 = Conv2D(
                in_channels=self.in_channels,
                out_channels=self.bottleneck_channels,
                kernel_size=1,
                scope="conv1")
            self.conv2 = Conv2D(
                in_channels=self.bottleneck_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                scope="conv2")

    def call(self, inputs):
        shortcut = inputs

        residual = self.conv1(inputs)
        residual = self.conv2(residual)
        output = shortcut + residual

        return slim.utils.collect_named_outputs(
            self.outputs_collections, self.sc.name, output)


@add_arg_scope
class DarkNetStage(Layer):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks,
        all_narrow=True,
        outputs_collections=None,
        **kwargs
    ):
        super(DarkNetStage, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            all_narrow=all_narrow,
            outputs_collections=outputs_collections,
            **kwargs)
        self.build()

    def build(self):
        self.blocks = []
        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:
            self.preconv = Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=2,
                scope="preconv"
            )
            block_channels = self.out_channels // 2 if self.all_narrow else self.out_channels
            bottleneck_channels = block_channels if self.all_narrow else block_channels // 2
            self.shortcut = Conv2D(
                in_channels=self.out_channels,
                out_channels=block_channels,
                kernel_size=1,
                scope="shortcut"
            )
            self.main = Conv2D(
                in_channels=self.out_channels,
                out_channels=block_channels,
                kernel_size=1,
                scope="main"
            )
            for i in range(self.num_blocks):
                self.blocks.append(
                    DarkNetResidualBlock(
                        in_channels=block_channels,
                        out_channels=block_channels,
                        bottleneck_channels=bottleneck_channels,
                        scope="block_{:d}".format(i + 1)
                    )
                )
            self.postconv = Conv2D(
                in_channels=block_channels,
                out_channels=block_channels,
                kernel_size=1,
                scope="postconv"
            )
            self.final = Conv2D(
                in_channels=block_channels * 2,
                out_channels=self.out_channels,
                kernel_size=1,
                scope="final"
            )

    def call(self, inputs):
        pre = self.preconv(inputs)
        shortcut = self.shortcut(pre)
        residual = self.main(pre)
        for block in self.blocks:
            residual = block(residual)
        post = self.postconv(residual)
        route = tf.concat([post, shortcut], axis=3)
        ret = self.final(route)
        return slim.utils.collect_named_outputs(self.outputs_collections, self.sc.name, ret)


@BACKBONE_REGISTRY.register()
class DarkNet53(Backbone): 

    def __init__(self, cfg, input_shape, **kwargs):

        norm = cfg.MODEL.RESNETS.NORM
        activation = cfg.MODEL.RESNETS.ACTIVATION
        freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

        stem_in_channels = input_shape.channels
        stem_out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        out_features = cfg.MODEL.RESNETS.OUT_FEATURES

        super(DarkNet53, self).__init__(
            norm=norm,
            activation=activation,
            freeze_at=freeze_at,
            stem_in_channels=stem_in_channels,
            stem_out_channels=stem_out_channels,
            res2_out_channels=res2_out_channels,
            **kwargs)
        self._out_features = out_features
        self.build()

    def build(self):
        out_stage_idx = [{"res1": 1, "res2": 2, "res3": 3, "res4": 4, "res5": 5}[f]
                         for f in self._out_features]
        max_stage_idx = max(out_stage_idx)
        num_blocks_per_stage = [1, 2, 8, 8, 4]
        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as sc:
            self.end_points_collection = sc.original_name_scope + '_end_points'
            with arg_scope(
                    [Conv2D, DarkNetResidualBlock, DarkNetStage],
                    outputs_collections=self.end_points_collection
            ):
                with darknet_arg_scope(self.freeze_at > 0, self.norm, self.activation):
                    self.stem = Conv2D(
                        in_channels=self.stem_in_channels,
                        out_channels=self.stem_out_channels,
                        kernel_size=3,
                        scope="stem"
                    )

                    current_stride = 1
                    self._out_feature_strides = {"stem": 1}
                    self._out_feature_channels = {"stem": self.stem_out_channels}

                in_channels = self.stem_out_channels
                out_channels = self.res2_out_channels
                self.stages = []
                for idx, stage_idx in enumerate(range(1, max_stage_idx + 1)):
                    all_narrow = False if stage_idx == 1 else True
                    with darknet_arg_scope(self.freeze_at >= stage_idx, self.norm, self.activation):
                        scope = "res" + str(stage_idx)
                        self.stages.append(
                            DarkNetStage(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                num_blocks=num_blocks_per_stage[idx],
                                all_narrow=all_narrow,
                                scope=scope
                            )
                        )
                    self._out_feature_strides[scope] = current_stride = int(current_stride * 2)
                    self._out_feature_channels[scope] = out_channels
                    in_channels = out_channels
                    out_channels *= 2

        if not self._out_features: self._out_features = [scope]

    def call(self, inputs):
        ret = self.stem(inputs)

        for stage in self.stages:
            ret = stage(ret)

        end_points = slim.utils.convert_collection_to_dict(self.end_points_collection)

        outputs = {}
        for name in end_points:
            last_name = name.split("/")[-1]
            if last_name in self._out_features:
                outputs[last_name] = end_points[name]

        assert len(outputs) == len(self._out_features), outputs
        return outputs
