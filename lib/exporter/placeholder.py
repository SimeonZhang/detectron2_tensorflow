import functools
import tensorflow as tf

from ..data import transform
from ..data import fields
from ..layers import resize_images


def build_batched_input_placeholder(cfg):
    """build a input placeholder composing of tensor_dict.

    Return:
        placeholder: a tensor_dict of preprocessed batched placeholders.
    """
    encoded_input_images = tf.placeholder(tf.string, [None])

    def preprocess_single_image(encoded_input_image):
        input_image = tf.image.decode_image(encoded_input_image, channels=3)
        input_image.set_shape([None, None, 3])

        min_dimension = cfg.TRANSFORM.RESIZE.MIN_SIZE_TEST
        max_dimension = cfg.TRANSFORM.RESIZE.MAX_SIZE_TEST

        orig_shape = tf.shape(input_image)[:2]
        resize_shape = transform.compute_new_shape(orig_shape, min_dimension, max_dimension)

        resized_input_image = resize_images(input_image, resize_shape, align_corners=True)
        resized_input_image = tf.pad(
            resized_input_image,
            [[0, max_dimension - resize_shape[0]], [0, max_dimension - resize_shape[1]], [0, 0]]
        )
        return resized_input_image, resize_shape

    resized_input_image, resized_input_shapes = tf.map_fn(
        preprocess_single_image, encoded_input_images, dtype=(tf.float32, tf.int32)
    )
    max_input_shape = tf.reduce_max(resized_input_shapes, axis=0)
    resized_input_image = resized_input_image[:, :max_input_shape[0], :max_input_shape[1], :]

    batched_inputs = {
        "image": resized_input_image,
        'image_shape': resized_input_shapes
    }
    return encoded_input_images, batched_inputs


def build_single_input_placeholder(
    cfg,
    encoded_image_name=None,
    decoded_image_name=None,
    expand_batch_dimension=False
):
    """build a input placeholder composing of tensor_dict.

    Return:
        placeholder: a tensor_dict of preprocessed batched placeholders.
    """
    encoded_input_image = tf.placeholder(tf.string, name=encoded_image_name)

    min_dimension = cfg.TRANSFORM.RESIZE.MIN_SIZE_TEST
    max_dimension = cfg.TRANSFORM.RESIZE.MAX_SIZE_TEST

    decoded_input_image = tf.image.decode_image(encoded_input_image, channels=3)
    decoded_input_image.set_shape([None, None, 3])
    if expand_batch_dimension:
        decoded_input_image = tf.expand_dims(
            decoded_input_image, axis=0, name=decoded_image_name)
        input_image = tf.squeeze(decoded_input_image, axis=0)
    elif decoded_image_name is not None:
        decoded_input_image = tf.identity(
            decoded_input_image, name=decoded_image_name)
        input_image = decoded_input_image

    orig_shape = tf.shape(input_image)[:2]
    resize_shape = transform.compute_new_shape(orig_shape, min_dimension, max_dimension)
    resized_input_image = resize_images(input_image, resize_shape, align_corners=True)

    batched_inputs = {
        "image": tf.expand_dims(resized_input_image, axis=0),
        'image_shape': tf.expand_dims(resize_shape, axis=0)
    }
    return encoded_input_image, decoded_input_image, batched_inputs