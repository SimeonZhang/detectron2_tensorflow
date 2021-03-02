import tensorflow as tf
from .base import Layer
from ..utils.arg_scope import add_arg_scope
from ..utils.shape_utils import combined_static_and_dynamic_shape

slim = tf.contrib.slim

__all__ = ["DropBlock"]


@add_arg_scope
class DropBlock(Layer):
    """DropBlock: a regularization method for convolutional neural networks.

    DropBlock is a form of structured dropout, where units in a contiguous
    region of a feature map are dropped together. DropBlock works better than
    dropout on convolutional layers due to the fact that activation units in
    convolutional layers are spatially correlated.
    See https://arxiv.org/pdf/1810.12890.pdf for details.
    """
    def __init__(self,
                 dropblock_keep_prob=None,
                 dropblock_size=None,
                 training=True,
                 **kwargs):
        super(DropBlock, self).__init__(
            dropblock_keep_prob=dropblock_keep_prob,
            dropblock_size=dropblock_size,
            training=training,
            **kwargs)

    def call(self, inputs):
        if (not self.training or self.dropblock_keep_prob is None or
                self.dropblock_keep_prob == 1.0):
            return inputs

        combined_shape = combined_static_and_dynamic_shape(inputs)
        height, width = combined_shape[1:3]

        total_size = height * width
        dropblock_size = tf.minimum(self.dropblock_size, tf.minimum(height, width))
        gamma = (1.0 - self.dropblock_keep_prob) * total_size / dropblock_size ** 2 / (
            (width - self.dropblock_size + 1) * (height - self.dropblock_size + 1)
        )

        # Force the block to be inside the feature map
        ind_w, ind_h = tf.meshgrid(tf.range(width), tf.range(height))
        is_valid_seed = tf.logical_and(
            tf.logical_and(
                ind_w >= tf.cast(dropblock_size // 2, tf.int32),
                ind_w < width - tf.cast((dropblock_size - 1) // 2, tf.int32)
            ),
            tf.logical_and(
                ind_h >= tf.cast(dropblock_size // 2, tf.int32),
                ind_h < height - tf.cast((dropblock_size - 1) // 2, tf.int32)
            )
        )
        is_valid_seed = tf.reshape(is_valid_seed, [1, height, width, 1])

        randnoise = tf.random_uniform(combined_shape, minval=0., maxval=1., dtype=tf.float32)
        is_valid_seed = tf.cast(is_valid_seed, tf.float32)
        seed_keep_rate = tf.cast(1. - gamma, tf.float32)
        block_pattern = (1. - is_valid_seed + seed_keep_rate + randnoise) >= 1.0
        block_pattern = tf.cast(block_pattern, tf.float32)

        kernel_size = [1, self.dropblock_size, self.dropblock_size, 1]
        block_pattern = -tf.nn.maxpool(
            -block_pattern,
            kernel_size=kernel_size,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        percent_ones = tf.cast(tf.reduce_sum(block_pattern), tf.float32) / tf.cast(
            tf.size(block_pattern), tf.float32
        )

        res = inputs / tf.cast(percent_ones, inputs.dtype) * tf.cast(block_pattern, inputs.dtype)
        return res
