import tensorflow as tf
from ..utils import shape_utils

slim = tf.contrib.slim

__all__ = ["subsample", "upsample", "flatten", "crop_and_resize"]


def resize_images(images, size, method="bilinear", **kwargs):
    """Take v2 image resize function as the first option, fall down to
    v1 function when failing.
    Args:
      images: 4-D Tensor of shape [batch, height, width, channels] or 3-D 
        Tensor of shape [height, width, channels].
      size: A 1-D int32 Tensor of 2 elements: new_height, new_width. 
        The new size for the images.
      method: ResizeMethod. Defaults to bilinear.
    Returns:
      images resized with given size.
    """
    try:
        resize_fn = tf.compat.v2.image.resize
        arg_names = ["preserve_aspect_ratio", "antialias", "name"]
        kwargs = {k: kwargs[k] for k in kwargs if k in arg_names}
    except:
        resize_fn = tf.image.resize_images
        methods = tf.image.ResizeMethod
        method = {
            "area": methods.AREA,
            "bicubic": methods.BICUBIC,
            "bilinear": methods.BILINEAR,
            "nearest": methods.NEAREST_NEIGHBOR
        }[method]
        arg_names = ["align_corners", "preserve_aspect_ratio", "name"]
        kwargs = {k: kwargs[k] for k in kwargs if k in arg_names}
    return resize_fn(images, size, method, **kwargs)


def subsample(inputs, factor, scope=None):
    """Subsample the input along the spatial dimensions.

    Args:
        inputs: A `Tensor` of size [batch, height_in, width_in, channels].
        factor: The subsampling factor.
        scope: Optional variable_scope.

    Returns:
        output: A `Tensor` of size [batch, height_out, width_out, channels]
        with the input, either intact (if factor == 1) or subsampled
        (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def upsample(inputs, factor, scope=None):
    """Upsample the input along the spatial dimensions.

    Args:
        inputs: A `Tensor` of size [batch, height_in, width_in, channels].
        factor: The upsampling factor.
        scope: Optional variable_scope.

    Returns:
        output: A `Tensor` of size [batch, height_out, width_out, channels]
        with the input, either intact (if factor == 1) or upsampled
        (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        with tf.name_scope(scope, "Upsample", [inputs]):
            batch_size, height, width, channels = (
                shape_utils.combined_static_and_dynamic_shape(inputs)
            )
            try:
                # resize v1 is not aligned
                resize = tf.compat.v2.image.resize
                return resize(
                    inputs, [height * factor, width * factor], 'nearest')
            except AttributeError:
                outputs = tf.reshape(
                    inputs,
                    [batch_size, height, 1, width, 1, channels]) * tf.ones(
                    [1, 1, factor, 1, factor, 1], dtype=inputs.dtype)
                return tf.reshape(
                    outputs, [batch_size, height * factor, width * factor, channels]
                )


def flatten(inputs, scope=None):
    if len(inputs.get_shape()) <= 2:
        return inputs
    else:
        return slim.flatten(inputs, scope=scope)


def crop_and_resize(
    image,
    boxes,
    box_ind,
    crop_size,
    aligned=True,
    method='bilinear',
    pad_border=True
):
    """
    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.

    Args:
        image: [n, h, w, c]
        boxes: [n, 4], ymin, xmin, ymax, xmax
        box_ind: [n]
        crop_size [2]:
    Returns:
        n,size,size,C
    """
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(image, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
        boxes = boxes + 1.

    def transform_fpcoor_for_tf(boxes, image_shape, crop_size):
        """
        This function transform fpcoor boxes to a format to be used by
        tf.image.crop_and_resize

        Returns:
            boxes
        """
        ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=1)

        if aligned:
            spacing_h = (ymax - ymin) / tf.cast(crop_size[0], tf.float32)
            spacing_w = (xmax - xmin) / tf.cast(crop_size[1], tf.float32)

            imshape = [
                tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)
            ]
            norm_ymin = (ymin + spacing_h / 2 - 0.5) / imshape[0]
            norm_xmin = (xmin + spacing_w / 2 - 0.5) / imshape[1]

            norm_h = spacing_h * tf.cast(crop_size[0] - 1, tf.float32) / imshape[0]
            norm_w = spacing_w * tf.cast(crop_size[1] - 1, tf.float32) / imshape[1]
            tf_boxes = tf.concat(
                [norm_ymin, norm_xmin, norm_ymin + norm_h, norm_xmin + norm_w], axis=1
            )
        else:
            imshape = [
                tf.cast(image_shape[0], tf.float32), tf.cast(image_shape[1], tf.float32)
            ]
            norm_ymin, norm_ymax = [y / imshape[0] for y in [ymin, ymax]]
            norm_xmin, norm_xmax = [x / imshape[1] for x in [xmin, xmax]]
            tf_boxes = tf.concat([norm_ymin, norm_xmin, norm_ymax, norm_xmax], axis=1)
        return tf_boxes

    image_shape = tf.shape(image)[1:3]
    boxes = transform_fpcoor_for_tf(boxes, image_shape, crop_size)
    ret = tf.image.crop_and_resize(
        image, boxes, tf.cast(box_ind, tf.int32), crop_size=crop_size, method=method)
    return ret


def drop_connect(inputs, is_training, drop_connect_rate):
    """Apply drop connect.
    Args:
        inputs: `Tensor` input tensor.
        is_training: `bool` if True, the model is in training mode.
        drop_connect_rate: `float` drop connect rate.
    Returns:
        A output tensor, which should have the same shape as input.
    """
    if not is_training or drop_connect_rate is None or drop_connect_rate == 0:
        return inputs

    keep_prob = 1.0 - drop_connect_rate
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.div(inputs, keep_prob) * binary_tensor
    return output
