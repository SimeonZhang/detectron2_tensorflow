import functools
import tensorflow as tf


def mish(inputs):
    with tf.name_scope("mish"):
        return inputs * tf.nn.tanh(tf.nn.softplus(inputs))


def get_activation(activation, **kwargs):
    if activation == "mish":
        return mish
    elif activation == "relu":
        return tf.nn.relu
    elif activation in ["leaky", "leaky_relu"]:
        return functools.partial(tf.nn.leaky_relu, **kwargs)
    elif activation in ["none", "linear", ""]:
        return
    else:
        raise ValueError("activation type {:s} not implemented!".format(activation))
