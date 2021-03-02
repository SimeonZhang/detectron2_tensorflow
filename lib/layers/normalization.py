import tensorflow as tf
from tensorflow.python.training import moving_averages

from .base import Layer
from ..utils import tf_utils
from ..utils import shape_utils
from ..utils.arg_scope import add_arg_scope

slim = tf.contrib.slim

__all__ = ["BatchNorm", "GroupNorm", "get_norm"]


@add_arg_scope
class BatchNorm(Layer):

    def __init__(self,
                 channels,
                 decay=0.9,
                 center=True,
                 scale=True,
                 epsilon=1e-5,
                 activation=None,
                 gamma_initializer=tf.ones_initializer(),
                 beta_initializer=tf.zeros_initializer(),
                 moving_mean_initializer=tf.zeros_initializer(),
                 moving_variance_initializer=tf.ones_initializer(),
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 variables_collections=None,
                 trainable=True,
                 outputs_collections=None,
                 fused=None,
                 sync=False,
                 **kwargs):
        super(BatchNorm, self).__init__(
            channels=channels,
            decay=decay,
            center=center,
            scale=scale,
            epsilon=epsilon,
            activation=activation,
            gamma_initializer=gamma_initializer,
            beta_initializer=beta_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            gamma_regularizer=gamma_regularizer,
            beta_regularizer=beta_regularizer,
            variables_collections=variables_collections,
            trainable=trainable,
            outputs_collections=outputs_collections,
            fused=fused,
            sync=sync,
            **kwargs)
        if not self.trainable and self.training:
            raise ValueError(
                "[BatchNorm] Frozen BatchNorm should not update EMA!")
        self.build()

    def build(self):
        params_shape = [self.channels]
        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:
            self.beta, self.gamma = None, None
            if self.center:
                self.beta = slim.model_variable(
                    "beta",
                    shape=params_shape,
                    dtype=self.dtype,
                    initializer=self.beta_initializer,
                    regularizer=self.beta_regularizer,
                    trainable=self.trainable,
                    collections=self.variables_collections)
            if self.scale:
                self.gamma = slim.model_variable(
                    "gamma",
                    shape=params_shape,
                    dtype=self.dtype,
                    initializer=self.gamma_initializer,
                    regularizer=self.gamma_regularizer,
                    trainable=self.trainable,
                    collections=self.variables_collections)
            self.moving_mean = slim.model_variable(
                "moving_mean",
                shape=params_shape,
                dtype=self.dtype,
                initializer=self.moving_mean_initializer,
                trainable=False,
                collections=self.variables_collections)
            self.moving_variance = slim.model_variable(
                "moving_variance",
                shape=params_shape,
                dtype=self.dtype,
                initializer=self.moving_variance_initializer,
                trainable=False,
                collections=self.variables_collections)

    def call(self, inputs):
        do_sync = self.sync and self.training
        if not do_sync:
            with tf.variable_scope(
                    self.sc, values=[inputs], reuse=True,
                    auxiliary_name_scope=False) as sc:
                layer = tf.layers.BatchNormalization(
                    momentum=self.decay,
                    epsilon=self.epsilon,
                    center=self.center,
                    scale=self.scale,
                    beta_initializer=self.beta_initializer,
                    gamma_initializer=self.gamma_initializer,
                    moving_mean_initializer=self.moving_mean_initializer,
                    moving_variance_initializer=self.moving_variance_initializer,
                    gamma_regularizer=self.gamma_regularizer,
                    beta_regularizer=self.beta_regularizer,
                    trainable=self.trainable,
                    name=sc.name,
                    dtype=inputs.dtype.base_dtype,
                    _scope=sc,
                    _reuse=True)
                ret = layer.apply(inputs, training=self.training)
        else:
            num_dev = len(tf_utils.get_available_gpus)
            if tf_utils.get_tf_version_tuple <= (1, 12):
                try:
                    from tensorflow.contrib.nccl.python.ops.nccl_ops \
                        import _validate_and_load_nccl_so
                except Exception:
                    pass
                else:
                    _validate_and_load_nccl_so()
                from tensorflow.contrib.nccl.ops import gen_nccl_ops
            else:
                from tensorflow.python.ops import gen_nccl_ops
            batch_mean = tf.reduce_mean(inputs, axis=[0, 1, 2])
            batch_mean_square = tf.reduce_mean(
                tf.square(inputs), axis=[0, 1, 2])
            shared_name = re.sub('tower[0-9]+/', '',
                                 tf.get_variable_scope().name)
            batch_mean = gen_nccl_ops.nccl_all_reduce(
                input=batch_mean,
                reduction='sum',
                num_devices=num_dev,
                shared_name=shared_name + '_NCCL_mean') * (1.0 / num_dev)
            batch_mean_square = gen_nccl_ops.nccl_all_reduce(
                input=batch_mean_square,
                reduction='sum',
                num_devices=num_dev,
                shared_name=shared_name + '_NCCL_var') * (1.0 / num_dev)
            batch_var = batch_mean_square - tf.square(batch_mean)

            ret = tf.nn.batch_normalization(inputs,
                                            mean=batch_mean,
                                            variance=batch_var,
                                            offset=self.beta,
                                            scale=self.gamma,
                                            variance_epsilon=self.epsilon)

            update_moving_mean = moving_averages.assign_moving_average(
                self.moving_mean, batch_mean,
                self.decay, zero_debias=False)
            update_moving_var = moving_averages.assign_moving_average(
                self.moving_variance, batch_var,
                self.decay, zero_debias=False)
            with tf.control_dependencies(
                    [update_moving_mean, update_moving_var]):
                ret = tf.identity(ret)

        if self.activation is not None:
            ret = activation(ret)
        return slim.utils.collect_named_outputs(
            self.outputs_collections, self.sc.name, ret)


@add_arg_scope
class GroupNorm(Layer):

    def __init__(self,
                 channels,
                 num_groups=32,
                 center=True,
                 scale=True,
                 epsilon=1e-5,
                 activation=None,
                 gamma_initializer=tf.ones_initializer(),
                 beta_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 variables_collections=None,
                 trainable=True,
                 outputs_collections=None,
                 **kwargs):
        if channels % num_groups != 0:
            raise ValueError(
                '"channels" {:d} is not divisible by "num_groups"'
                ' {:d}.'.format(self.channels, self.num_groups))
        super(GroupNorm, self).__init__(
            channels=channels,
            num_groups=num_groups,
            center=center,
            scale=scale,
            epsilon=epsilon,
            activation=activation,
            gamma_initializer=gamma_initializer,
            beta_initializer=beta_initializer,
            gamma_regularizer=gamma_regularizer,
            beta_regularizer=beta_regularizer,
            variables_collections=variables_collections,
            trainable=trainable,
            outputs_collections=outputs_collections,
            **kwargs)
        self.build()

    def build(self):
        params_shape = [self.channels]
        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:
            self.beta, self.gamma = None, None
            if self.center:
                self.beta = slim.model_variable(
                    "beta",
                    shape=params_shape,
                    dtype=self.dtype,
                    initializer=self.beta_initializer,
                    regularizer=self.beta_regularizer,
                    trainable=self.trainable,
                    collections=self.variables_collections)
            if self.scale:
                self.gamma = slim.model_variable(
                    "gamma",
                    shape=params_shape,
                    dtype=self.dtype,
                    initializer=self.gamma_initializer,
                    regularizer=self.gamma_regularizer,
                    trainable=self.trainable,
                    collections=self.variables_collections)

    def call(self, inputs):
        orig_shape = shape_utils.combined_static_and_dynamic_shape(inputs)
        assert len(orig_shape) == 4
        batch, height, width, channels = orig_shape
        assert channels == self.channels
        group_size = self.channels // self.num_groups

        inputs = tf.reshape(inputs,
                            [batch, height, width,
                             self.num_groups, group_size])
        mean, var = tf.nn.moments(inputs, [1, 2, 4], keep_dims=True)

        new_shape = [1, 1, 1, self.num_groups, group_size]
        gamma = tf.reshape(self.gamma, new_shape)
        beta = tf.reshape(self.beta, new_shape)

        ret = tf.nn.batch_normalization(inputs,
                                        mean=mean,
                                        variance=var,
                                        offset=beta,
                                        scale=gamma,
                                        variance_epsilon=self.epsilon)
        ret = tf.reshape(ret, orig_shape)
        if self.activation is not None:
            ret = activation(ret)
        return slim.utils.collect_named_outputs(
            self.outputs_collections, self.sc.name, ret)


def get_norm(norm):
    if isinstance(norm, str):
        if len(norm) == 0:
            norm = None
        elif norm == "GN":
            norm = GroupNorm
        elif norm in ["BN", "FrozenBN", "SyncBN"]:
            norm = BatchNorm
        else:
            raise ValueError("{:s} is not recognized !".format(norm))
    return norm
