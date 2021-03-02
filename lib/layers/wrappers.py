import tensorflow as tf

from .base import Layer
from .functional import upsample
from .convolutional import fix_padding
from ..utils import shape_utils
from ..utils.arg_scope import add_arg_scope

slim = tf.contrib.slim


@add_arg_scope
class Linear(Layer):

    def __init__(self,
                 in_units,
                 out_units,
                 activation=None,
                 normalizer=None,
                 normalizer_params=None,
                 use_bias=True,
                 weights_initializer=None,
                 weights_regularizer=None,
                 bias_initializer=tf.zeros_initializer(),
                 bias_regularizer=None,
                 variables_collections=None,
                 trainable=True,
                 outputs_collections=None,
                 **kwargs):
        super(Linear, self).__init__(
            in_units=in_units,
            out_units=out_units,
            activation=activation,
            normalizer=normalizer,
            normalizer_params=normalizer_params,
            use_bias=use_bias,
            weights_initializer=weights_initializer,
            weights_regularizer=weights_regularizer,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            variables_collections=variables_collections,
            trainable=trainable,
            outputs_collections=outputs_collections,
            **kwargs)
        self.build()

    def build(self):
        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:
            self.weights = slim.model_variable(
                "weights",
                shape=[self.in_units, self.out_units],
                dtype=self.dtype,
                initializer=self.weights_initializer,
                regularizer=self.weights_regularizer,
                trainable=self.trainable,
                collections=self.variables_collections)
            if self.use_bias:
                self.bias = slim.model_variable(
                    "bias",
                    shape=[self.out_units],
                    dtype=self.dtype,
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    trainable=self.trainable,
                    collections=self.variables_collections)

            if self.normalizer is not None:
                normalizer_params = self.normalizer_params or {}
                self.normalizer_fn = normalizer(**normalizer_params)

    def call(self, inputs):
        layer_variable_getter = self.build_variable_getter({'kernel': 'weights'})
        with tf.variable_scope(
                self.sc, values=[inputs], reuse=True,
                custom_getter=layer_variable_getter,
                auxiliary_name_scope=False) as sc:
            layer = tf.layers.Dense(
                units=self.out_units,
                activation=None,
                use_bias=self.use_bias,
                kernel_initializer=self.weights_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.weights_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=None,
                trainable=self.trainable,
                name=sc.name,
                dtype=inputs.dtype.base_dtype,
                _scope=sc,
                _reuse=True)
            ret = layer.apply(inputs)

        if self.normalizer is not None:
            ret = self.normalizer_fn(ret)

        if self.activation is not None:
            ret = self.activation(ret)

        return slim.utils.collect_named_outputs(
            self.outputs_collections, self.sc.name, ret)


@add_arg_scope
class Upsample(Layer):

    def __init__(
        self,
        factor,
        **kwargs,
    ):
        super().__init__(factor=factor, **kwargs)

    def call(self, x):
        ret = upsample(x, self.factor)
        return ret


@add_arg_scope
class MaxPool2D(Layer):

    def __init__(self, kernel_size, stride=1, padding="SAME", **kwargs):
        super().__init__(
            kernel_size=kernel_size, stride=stride, padding=padding, **kwargs
        )

    def call(self, x):
        x = fix_padding(x, self.kernel_size, padding=self.padding)
        x = slim.max_pool2d(
            x, kernel_size=self.kernel_size, stride=self.stride, padding="VALID"
        )
        return x
