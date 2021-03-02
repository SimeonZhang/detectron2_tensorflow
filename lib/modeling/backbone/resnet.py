from contextlib import ExitStack, contextmanager
import copy
import tensorflow as tf

from ...layers import (
    Layer,
    Conv2D,
    DeformConv2D,
    ModulatedDeformConv2D,
    BatchNorm,
    ShapeSpec,
    get_norm
)
from ...utils.arg_scope import add_arg_scope, arg_scope
from .blocks import BaseBlock, BottleneckBlock, DeformBottleneckBlock
from .backbone import Backbone
from .build import BACKBONE_REGISTRY

slim = tf.contrib.slim


@contextmanager
def resnet_arg_scope(freeze, norm):
    normalizer = get_norm(norm)
    with arg_scope(
        [Conv2D, DeformConv2D, ModulatedDeformConv2D],
        use_bias=False,
        normalizer=normalizer,
        normalizer_params={"scope": "norm"},
        activation=tf.nn.relu,
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
            stack.enter_context(
                arg_scope(
                    [Conv2D, DeformConv2D, ModulatedDeformConv2D, BatchNorm], trainable=False
                )
            )

        yield


@add_arg_scope
class Stem(Layer):

    def __init__(
        self,
        in_channels,
        out_channels,
        outputs_collections=None,
        **kwargs
    ):
        super(Stem, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            outputs_collections=outputs_collections,
            **kwargs)
        self.build()

    def build(self):
        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:
            self.conv1 = Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=7,
                stride=2,
                normalizer_params={"channels": self.out_channels, "scope": "norm"},
                scope="conv1")

    def call(self, inputs):
        ret = self.conv1(inputs)
        ret = tf.pad(ret, [[0, 0], [1, 1], [1, 1], [0, 0]])
        ret = slim.max_pool2d(ret, [3, 3], stride=2, padding='VALID', scope='pool1')
        return slim.utils.collect_named_outputs(
            self.outputs_collections, self.sc.name, ret)

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool


@add_arg_scope
class Stage(Layer):

    def __init__(
        self,
        block_class,
        block_kwargs,
        num_blocks,
        first_stride,
        outputs_collections=None,
        **kwargs
    ):
        assert issubclass(block_class, BaseBlock), block_class
        super(Stage, self).__init__(
            block_class=block_class,
            block_kwargs=block_kwargs,
            num_blocks=num_blocks,
            first_stride=first_stride,
            outputs_collections=outputs_collections,
            **kwargs)
        self.build()

    def build(self):
        self.blocks = []
        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:
            for i in range(self.num_blocks):
                block_kwargs = copy.copy(self.block_kwargs)
                block_kwargs["scope"] = 'block_{:d}'.format(i + 1)
                if i == 0:
                    block_kwargs["stride"] = self.first_stride
                else:
                    block_kwargs["stride"] = 1
                    block_kwargs["in_channels"] = self.block_kwargs["out_channels"]
                self.blocks.append(self.block_class(**block_kwargs))

    def call(self, inputs):
        ret = inputs
        for block in self.blocks:
            ret = block(ret)
        return slim.utils.collect_named_outputs(self.outputs_collections, self.sc.name, ret)


@BACKBONE_REGISTRY.register()
class ResNet(Backbone):

    def __init__(self, cfg, input_shape, **kwargs):

        norm = cfg.MODEL.RESNETS.NORM
        depth = cfg.MODEL.RESNETS.DEPTH
        freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

        stem_in_channels = input_shape.channels
        stem_out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        out_features = cfg.MODEL.RESNETS.OUT_FEATURES

        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION
        deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
        deform_modulated = cfg.MODEL.RESNETS.DEFORM_MODULATED
        deform_num_groups = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS

        if res5_dilation not in [1, 2]:
            raise ValueError(
                "res5_dilation cannot be {}.".format(res5_dilation))
        super(ResNet, self).__init__(
            norm=norm,
            depth=depth,
            freeze_at=freeze_at,
            stem_in_channels=stem_in_channels,
            stem_out_channels=stem_out_channels,
            res2_out_channels=res2_out_channels,
            num_groups=num_groups,
            width_per_group=width_per_group,
            stride_in_1x1=stride_in_1x1,
            res5_dilation=res5_dilation,
            deform_on_per_stage=deform_on_per_stage,
            deform_modulated=deform_modulated,
            deform_num_groups=deform_num_groups,
            **kwargs)
        self._out_features = out_features
        self.build()

    def build(self):
        out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f]
                         for f in self._out_features]
        max_stage_idx = max(out_stage_idx)
        num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3],
                                152: [3, 8, 36, 3]}[self.depth]
        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as sc:
            self.end_points_collection = sc.original_name_scope + '_end_points'
            with arg_scope(
                    [Conv2D, DeformConv2D, ModulatedDeformConv2D,
                     BottleneckBlock, DeformBottleneckBlock, Stem, Stage],
                    outputs_collections=self.end_points_collection):
                with resnet_arg_scope(self.freeze_at > 0, self.norm):
                    self.stem = Stem(
                        in_channels=self.stem_in_channels,
                        out_channels=self.stem_out_channels,
                        scope="stem"
                    )
                    current_stride = self.stem.stride
                    self._out_feature_strides = {"stem": current_stride}
                    self._out_feature_channels = {"stem": self.stem.out_channels}

                in_channels = self.stem_out_channels
                out_channels = self.res2_out_channels
                bottleneck_channels = self.num_groups * self.width_per_group
                self.stages = []
                for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
                    rate = self.res5_dilation if stage_idx == 5 else 1
                    first_stride = 2
                    if idx == 0 or (stage_idx == 5 and rate == 2):
                        first_stride = 1
                    block_kwargs = {
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "bottleneck_channels": bottleneck_channels,
                        "rate": rate,
                        "num_groups": self.num_groups,
                        "stride_in_1x1": self.stride_in_1x1
                    }
                    if self.deform_on_per_stage[idx]:
                        block_class = DeformBottleneckBlock
                        block_kwargs.update({
                            "deform_modulated": self.deform_modulated,
                            "deform_num_groups": self.deform_num_groups})
                    else:
                        block_class = BottleneckBlock
                    with resnet_arg_scope(self.freeze_at >= stage_idx, self.norm):
                        scope = "res" + str(stage_idx)
                        self.stages.append(
                            Stage(block_class=block_class,
                                  block_kwargs=block_kwargs,
                                  num_blocks=num_blocks_per_stage[idx],
                                  first_stride=first_stride,
                                  scope=scope))
                    self._out_feature_strides[scope] = current_stride = int(
                        current_stride * first_stride)
                    self._out_feature_channels[scope] = out_channels
                    in_channels = out_channels
                    out_channels *= 2
                    bottleneck_channels *= 2

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
