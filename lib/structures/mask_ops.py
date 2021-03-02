import tensorflow as tf

from . import box_list
from . import box_list_ops


def reframe_box_masks_to_image_masks(
    box_masks, boxes, image_shape, mask_threshold=0.5, scope=None
):
    """Transforms the box masks back to full image masks.
    Embeds masks in bounding boxes of larger masks whose shapes correspond to
    image shape.
    Args:
        box_masks: box masks.
        boxes: normalized boxes.
        mask_field: mask field name. 
        image_shape: The output mask will have the same shape as the image shape.
    Returns:
        image_masks: SparseBoxList containing reframed masks..
    """
    with tf.name_scope("PasteMasks", scope):
        boxlist = box_list.BoxList(boxes)
        boxes = box_list_ops.to_normalized_coordinates(boxlist, image_shape).boxes

        image_masks = tf.cond(
            tf.shape(box_masks)[0] > 0,
            lambda: reframe_box_masks_to_image_masks_default(box_masks, boxes, image_shape),
            lambda: tf.zeros([0, image_shape[0], image_shape[1], 1], dtype=tf.float32)
        )
        image_masks = tf.squeeze(image_masks, axis=3)
        image_masks = tf.cast(tf.greater(image_masks, mask_threshold), tf.uint8)
        return image_masks


def reframe_box_masks_to_image_masks_default(box_masks, boxes, image_shape):
    """The default function when there are more than 0 box masks."""
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
        with tf.name_scope("TransformBoxes"):
            boxes = tf.reshape(boxes, [-1, 2, 2])
            min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
            max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
            transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
            return tf.reshape(transformed_boxes, [-1, 4])

    box_masks = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(boxes)[0]
    unit_boxes = tf.concat(
        [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
    return tf.image.crop_and_resize(
        image=box_masks,
        boxes=reverse_boxes,
        box_ind=tf.range(num_boxes),
        crop_size=image_shape,
        method="bilinear",
        extrapolation_value=0.0)

