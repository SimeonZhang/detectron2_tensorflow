import numpy as np
import tensorflow as tf

from ...layers import Layer, Conv2D, Linear, ShapeSpec, get_norm, flatten
from ...utils import registry
from ...utils import tf_utils
from ...utils.arg_scope import arg_scope

ROI_BOX_HEAD_REGISTRY = registry.Registry("ROI_BOX_HEAD")
ROI_BOX_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.
The registered object will be called with `obj(cfg, input_shape)`.
"""


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHead(Layer):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec, **kwargs):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super(FastRCNNConvFCHead, self).__init__(**kwargs)

        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm = cfg.MODEL.ROI_BOX_HEAD.NORM

        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            normalizer = get_norm(norm)
            self.convs = []
            with arg_scope(
                    [Conv2D],
                    out_channels=conv_dim,
                    kernel_size=3,
                    use_bias=not normalizer,
                    normalizer=normalizer,
                    normalizer_params={"channels": conv_dim, "scope": "norm"},
                    activation=tf.nn.relu,
                    padding="SAME",
                    weights_initializer=tf.variance_scaling_initializer(
                        scale=2.0, mode='fan_out',
                        distribution='untruncated_normal' if tf_utils.get_tf_version_tuple() >= (1, 12) else 'normal')):
                for k in range(num_conv):
                    self.convs.append(
                        Conv2D(
                            in_channels=self._output_size[0], scope="conv{}".format(k + 1)
                        )
                    )
                    self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

            self.fcs = []
            with arg_scope(
                    [Linear],
                    out_units=fc_dim,
                    activation=tf.nn.relu,
                    weights_initializer=tf.variance_scaling_initializer()):
                for k in range(num_fc):
                    self.fcs.append(
                        Linear(np.prod(self._output_size), scope="fc{}".format(k + 1))
                    )
                    self._output_size = fc_dim

    def call(self, x):
        for layer in self.convs:
            x = layer(x)
        if len(self.fcs):
            if x.shape.ndims > 2:
                x = flatten(x)
            for layer in self.fcs:
                x = layer(x)
        return x

    @property
    def output_size(self):
        return self._output_size


def build_box_head(cfg, input_shape, **kwargs):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape, **kwargs)
