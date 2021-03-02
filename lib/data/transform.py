"""Preprocess images and bounding boxes for detection.

We perform two sets of operations in preprocessing stage:
(a) operations that are applied to both training and testing data,
(b) operations that are applied only to training data for the purpose of
    data augmentation.

A preprocessing function receives a set of inputs,
e.g. an image and bounding boxes,
performs an operation on them, and returns them.
Some examples are: randomly cropping the image, randomly mirroring the image,
                   randomly changing the brightness, contrast, hue and
                   randomly jittering the bounding boxes.

The preprocess function receives a tensor_dict which is a dictionary that maps
different field names to their tensors. For example,
tensor_dict[fields.InputFields.image] holds the image tensor.
The image is a rank 4 tensor: [1, height, width, channels] with
dtype=tf.float32. The gt_boxes is a rank 2 tensor: [N, 4] where
in each row there is a box with [ymin, xmin, ymax, xmax].
Boxes are in normalized coordinates meaning
their coordinate values range in [0, 1]

"""
import sys
import random
import tensorflow as tf

from ..structures import box_list, box_list_ops
from ..layers import resize_images
from . import fields

# -----------------------------------------------------------------
# FLIP / ROTATE
# -----------------------------------------------------------------


def _flip_boxes_left_right(boxes):
    """Left-right flip the boxes.

    Args:
        boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].

    Returns:
        Flipped boxes.
    """
    ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
    flipped_xmin = tf.subtract(1.0, xmax)
    flipped_xmax = tf.subtract(1.0, xmin)
    flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
    return flipped_boxes


def _flip_boxes_up_down(boxes):
    """Up-down flip the boxes.

    Args:
        boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].

    Returns:
        Flipped boxes.
    """
    ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
    flipped_ymin = tf.subtract(1.0, ymax)
    flipped_ymax = tf.subtract(1.0, ymin)
    flipped_boxes = tf.concat([flipped_ymin, xmin, flipped_ymax, xmax], 1)
    return flipped_boxes


def _rot90_boxes(boxes):
    """Rotate boxes counter-clockwise by 90 degrees.

    Args:
        boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].

    Returns:
        Rotated boxes.
    """
    ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
    rotated_ymin = tf.subtract(1.0, xmax)
    rotated_ymax = tf.subtract(1.0, xmin)
    rotated_xmin = ymin
    rotated_xmax = ymax
    rotated_boxes = tf.concat(
        [rotated_ymin, rotated_xmin, rotated_ymax, rotated_xmax], 1)
    return rotated_boxes


def _rot90_boxes_clockwise(boxes):
    """Rotate boxes counter-clockwise by 90 degrees.

    Args:
        boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].

    Returns:
        Rotated boxes.
    """
    ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
    rotated_ymin = xmin
    rotated_ymax = xmax
    rotated_xmin = tf.subtract(1.0, ymax)
    rotated_xmax = tf.subtract(1.0, ymin)

    rotated_boxes = tf.concat(
        [rotated_ymin, rotated_xmin, rotated_ymax, rotated_xmax], 1)
    return rotated_boxes


def _flip_masks_left_right(masks):
    """Left-right flip masks.

    Args:
        masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.

    Returns:
        flipped masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.
    """
    return masks[:, :, ::-1]


def _flip_masks_up_down(masks):
    """Up-down flip masks.

    Args:
        masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.

    Returns:
        flipped masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.
    """
    return masks[:, ::-1, :]


def _rot90_masks(masks):
    """Rotate masks counter-clockwise by 90 degrees.

    Args:
        masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.

    Returns:
        rotated masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.
    """
    masks = tf.transpose(masks, [0, 2, 1])
    return masks[:, ::-1, :]

def _rot90_masks_clockwise(masks):
    """Rotate masks counter-clockwise by 90 degrees.

    Args:
        masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.

    Returns:
        rotated masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.
    """
    masks = tf.transpose(masks, [0, 2, 1])
    return masks[:, :, ::-1]


def _flip_seg_left_right(sem_seg):
    """Left-right flip masks.

    Args:
        masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.

    Returns:
        flipped masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.
    """
    return sem_seg[:, ::-1]


def _flip_seg_up_down(sem_seg):
    """Up-down flip masks.

    Args:
        masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.

    Returns:
        flipped masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.
    """
    return sem_seg[::-1, :]


def _rot90_seg(sem_seg):
    """Rotate masks counter-clockwise by 90 degrees.

    Args:
        masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.

    Returns:
        rotated masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.
    """
    sem_seg = tf.transpose(sem_seg, [1, 0])
    return sem_seg[::-1, :]

def _rot90_seg_clockwise(sem_seg):
    """Rotate masks counter-clockwise by 90 degrees.

    Args:
        masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.

    Returns:
        rotated masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.
    """
    sem_seg = tf.transpose(sem_seg, [1, 0])
    return sem_seg[:, ::-1]


def random_horizontal_flip(image,
                           boxes=None,
                           masks=None,
                           sem_seg=None,
                           seed=None):
    """Randomly flips the image and detections horizontally.

    The probability of flipping the image is 50%.

    Args:
        image: rank 3 float32 tensor with shape [height, width, channels].
        boxes: (optional) rank 2 float32 tensor with shape [N, 4]
            containing the bounding boxes.
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].
        masks: (optional) rank 3 float32 tensor with shape
            [num_instances, height, width] containing instance masks. The masks
            are of the same height, width as the input `image`.
        seed: random seed

    Returns:
        image: image which is the same shape as input image.

        If boxes, masks, keypoints, and keypoint_flip_permutation are not None,
        the function also returns the following tensors.

        boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
        masks: rank 3 float32 tensor with shape [num_instances, height, width]
            containing instance masks.

    """

    def _flip_image(image):
        # flip image
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped

    with tf.name_scope('RandomHorizontalFlip', values=[image, boxes]):
        result = []
        # random variable defining whether to do flip or not
        do_a_flip_random = tf.greater(tf.random_uniform([], seed=seed), 0.5)

        # flip image
        image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
        result.append(image)

        # flip boxes
        if boxes is not None:
            boxes = tf.cond(do_a_flip_random, lambda: _flip_boxes_left_right(boxes),
                            lambda: boxes)
            result.append(boxes)

        # flip masks
        if masks is not None:
            masks = tf.cond(do_a_flip_random, lambda: _flip_masks_left_right(masks),
                            lambda: masks)
            result.append(masks)

        if sem_seg is not None:
            sem_seg = tf.cond(do_a_flip_random, lambda: _flip_seg_left_right(sem_seg),
                              lambda: sem_seg)
            result.append(sem_seg)

        return tuple(result)


def random_vertical_flip(image,
                         boxes=None,
                         masks=None,
                         sem_seg=None,
                         seed=None):
    """Randomly flips the image and detections vertically.

    The probability of flipping the image is 50%.

    Args:
        image: rank 3 float32 tensor with shape [height, width, channels].
        boxes: (optional) rank 2 float32 tensor with shape [N, 4]
            containing the bounding boxes.
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].
        masks: (optional) rank 3 float32 tensor with shape
            [num_instances, height, width] containing instance masks. The masks
            are of the same height, width as the input `image`.
        seed: random seed

    Returns:
        image: image which is the same shape as input image.

        If boxes, masks, keypoints, and keypoint_flip_permutation are not None,
        the function also returns the following tensors.

        boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
        masks: rank 3 float32 tensor with shape [num_instances, height, width]
            containing instance masks.

    """

    def _flip_image(image):
        # flip image
        image_flipped = tf.image.flip_up_down(image)
        return image_flipped

    with tf.name_scope('RandomVerticalFlip', values=[image, boxes]):
        result = []
        # random variable defining whether to do flip or not
        do_a_flip_random = tf.greater(tf.random_uniform([], seed=seed), 0.5)

        # flip image
        image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
        result.append(image)

        # flip boxes
        if boxes is not None:
            boxes = tf.cond(do_a_flip_random, lambda: _flip_boxes_up_down(boxes),
                            lambda: boxes)
            result.append(boxes)

        # flip masks
        if masks is not None:
            masks = tf.cond(do_a_flip_random, lambda: _flip_masks_up_down(masks),
                            lambda: masks)
            result.append(masks)

        if sem_seg is not None:
            sem_seg = tf.cond(do_a_flip_random, lambda: _flip_seg_up_down(sem_seg),
                              lambda: sem_seg)
            result.append(sem_seg)

        return tuple(result)


def random_rotation90(image,
                      boxes=None,
                      masks=None,
                      sem_seg=None,
                      seed=None):
    """Randomly rotates the image and detections 90 degrees counter-clockwise.

    The probability of rotating the image is 50%. This can be combined with
    random_horizontal_flip and random_vertical_flip to produce an output with a
    uniform distribution of the eight possible 90 degree rotation / reflection
    combinations.

    Args:
        image: rank 3 float32 tensor with shape [height, width, channels].
        boxes: (optional) rank 2 float32 tensor with shape [N, 4]
            containing the bounding boxes.
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].
        masks: (optional) rank 3 float32 tensor with shape
            [num_instances, height, width] containing instance masks. The masks
            are of the same height, width as the input `image`.
        seed: random seed

    Returns:
        image: image which is the same shape as input image.

        If boxes, masks, and keypoints, are not None,
        the function also returns the following tensors.

        boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
        masks: rank 3 float32 tensor with shape [num_instances, height, width]
            containing instance masks.
    """

    def _rot90_image(image):
        # flip image
        image_rotated = tf.image.rot90(image)
        return image_rotated

    with tf.name_scope('RandomRotation90', values=[image, boxes]):
        result = []

        # random variable defining whether to rotate by 90 degrees or not
        do_a_rot90_random = tf.greater(tf.random_uniform([], seed=seed), 0.5)

        # flip image
        image = tf.cond(do_a_rot90_random, lambda: _rot90_image(image),
                        lambda: image)
        result.append(image)

        # flip boxes
        if boxes is not None:
            boxes = tf.cond(do_a_rot90_random, lambda: _rot90_boxes(boxes),
                            lambda: boxes)
            result.append(boxes)

        # flip masks
        if masks is not None:
            masks = tf.cond(do_a_rot90_random, lambda: _rot90_masks(masks),
                            lambda: masks)
            result.append(masks)

        if sem_seg is not None:
            sem_seg = tf.cond(do_a_rot90_random, lambda: _rot90_seg(sem_seg),
                              lambda: sem_seg)
            result.append(sem_seg)

        return tuple(result)


def random_rotation90_both_direction(image,
                      boxes=None,
                      masks=None,
                      sem_seg=None,
                      seed=None):
    """Randomly rotates the image and detections 90 degrees counter-clockwise.

    The probability of rotating the image is 50%. This can be combined with
    random_horizontal_flip and random_vertical_flip to produce an output with a
    uniform distribution of the eight possible 90 degree rotation / reflection
    combinations.

    Args:
        image: rank 3 float32 tensor with shape [height, width, channels].
        boxes: (optional) rank 2 float32 tensor with shape [N, 4]
            containing the bounding boxes.
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].
        masks: (optional) rank 3 float32 tensor with shape
            [num_instances, height, width] containing instance masks. The masks
            are of the same height, width as the input `image`.
        seed: random seed

    Returns:
        image: image which is the same shape as input image.

        If boxes, masks, and keypoints, are not None,
        the function also returns the following tensors.

        boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
        masks: rank 3 float32 tensor with shape [num_instances, height, width]
            containing instance masks.
    """

    def _rot90_image(image):
        image_rotated = tf.image.rot90(image)
        return image_rotated

    def _rot90_image_clockwise(image):
        image_rotated = tf.image.rot90(image, k=3)
        return image_rotated

    with tf.name_scope('RandomRotationBoth90', values=[image, boxes]):
        result = []

        # random variable defining whether to rotate by 90 degrees or not
        prob = tf.random_uniform([], seed=seed)
        do_rotation = tf.greater(prob, 0.5)
        do_clockwise = tf.greater(prob, 0.75)

        # rotate image
        image = tf.cond(do_rotation,
                        lambda: tf.cond(do_clockwise,
                                        lambda: _rot90_image_clockwise(image),
                                        lambda: _rot90_image(image)),
                        lambda: image)
        result.append(image)

        # rotate boxes
        if boxes is not None:
            boxes = tf.cond(do_rotation,
                            lambda: tf.cond(do_clockwise,
                                            lambda: _rot90_boxes_clockwise(boxes),
                                            lambda: _rot90_boxes(boxes)),
                            lambda: boxes)
            result.append(boxes)

        # rotate masks
        if masks is not None:
            masks = tf.cond(do_rotation,
                            lambda: tf.cond(do_clockwise,
                                             lambda: _rot90_masks_clockwise(masks),
                                             lambda: _rot90_masks(masks)),
                            lambda: masks)
            result.append(masks)

        if sem_seg is not None:
            sem_seg = tf.cond(do_rotation,
                              lambda: tf.cond(do_clockwise,
                                              lambda: _rot90_seg_clockwise(sem_seg),
                                              lambda: _rot90_seg(sem_seg)),
                              lambda: sem_seg)
            result.append(sem_seg)

        return tuple(result)
# -----------------------------------------------------------------
# PIXEL VALUE SCALE
# -----------------------------------------------------------------


def random_pixel_value_scale(image, minval=0.9, maxval=1.1, seed=None):
    """Scales each value in the pixels of the image.

        This function scales each pixel independent of the other ones.
        For each value in image tensor, draws a random number between
        minval and maxval and multiples the values with them.

    Args:
        image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
            with pixel values varying between [0, 1].
        minval: lower ratio of scaling pixel values.
        maxval: upper ratio of scaling pixel values.
        seed: random seed.

    Returns:
        image: image which is the same shape as input image.
    """
    with tf.name_scope('RandomPixelValueScale', values=[image]):
        color_coef = tf.random_uniform(
            tf.shape(image),
            minval=minval,
            maxval=maxval,
            dtype=tf.float32,
            seed=seed)
        image = tf.multiply(image, color_coef)
        image = tf.clip_by_value(image, 0.0, 255.0)

    return image

# -----------------------------------------------------------------
# COLOR DISTORTION
# -----------------------------------------------------------------


def random_adjust_brightness(image, max_delta=0.2):
    """Randomly adjusts brightness.

    Makes sure the output image is still between 0 and 1.

    Args:
        image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
            with pixel values varying between [0, 1].
        max_delta: how much to change the brightness. A value between [0, 1).

    Returns:
        image: image which is the same shape as input image.
        boxes: boxes which is the same shape as input boxes.
    """
    with tf.name_scope('RandomAdjustBrightness', values=[image]):
        image = tf.image.random_brightness(image, max_delta)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
        return image


def random_adjust_contrast(image, min_delta=0.8, max_delta=1.25):
    """Randomly adjusts contrast.

    Makes sure the output image is still between 0 and 1.

    Args:
        image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
            with pixel values varying between [0, 1].
        min_delta: see max_delta.
        max_delta: how much to change the contrast. Contrast will change with a
                value between min_delta and max_delta. This value will be
                multiplied to the current contrast of the image.

    Returns:
        image: image which is the same shape as input image.
    """
    with tf.name_scope('RandomAdjustContrast', values=[image]):
        image = tf.image.random_contrast(image, min_delta, max_delta)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
        return image


def random_adjust_hue(image, max_delta=0.02):
    """Randomly adjusts hue.

    Makes sure the output image is still between 0 and 1.

    Args:
        image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
            with pixel values varying between [0, 1].
        max_delta: change hue randomly with a value between 0 and max_delta.

    Returns:
        image: image which is the same shape as input image.
    """
    with tf.name_scope('RandomAdjustHue', values=[image]):
        image = tf.image.random_hue(image, max_delta)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
        return image


def random_adjust_saturation(image, min_delta=0.8, max_delta=1.25):
    """Randomly adjusts saturation.

    Makes sure the output image is still between 0 and 1.

    Args:
        image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
            with pixel values varying between [0, 1].
        min_delta: see max_delta.
        max_delta: how much to change the saturation. Saturation will change with a
                value between min_delta and max_delta. This value will be
                multiplied to the current saturation of the image.

    Returns:
        image: image which is the same shape as input image.
    """
    with tf.name_scope('RandomAdjustSaturation', values=[image]):
        image = tf.image.random_saturation(image, min_delta, max_delta)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
        return image


def random_distort_color(image, color_ordering=0):
    """Randomly distorts color.

    Randomly distorts color using a combination of brightness, hue, contrast
    and saturation changes. Makes sure the output image is still between 0 and 1.

    Args:
        image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
            with pixel values varying between [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0, 1).

    Returns:
        image: image which is the same shape as input image.

    Raises:
        ValueError: if color_ordering is not in {0, 1}.
    """
    with tf.name_scope('RandomDistortColor', values=[image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
            image = tf.image.random_hue(image, max_delta=0.2)
        else:
            raise ValueError('color_ordering must be in {0, 1}')

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 255.0)
        return image

# -----------------------------------------------------------------
# JITTER BOXES
# -----------------------------------------------------------------


def random_jitter_boxes(boxes, ratio=0.05, seed=None):
    """Randomly jitter boxes in image.

    Args:
        boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].
        ratio: The ratio of the box width and height that the corners can jitter.
            For example if the width is 100 pixels and ratio is 0.05,
            the corners can jitter up to 5 pixels in the x direction.
        seed: random seed.

    Returns:
        boxes: boxes which is the same shape as input boxes.
    """
    def random_jitter_box(box, ratio, seed):
        """Randomly jitter box.

        Args:
            box: bounding box [1, 1, 4].
            ratio: max ratio between jittered box and original box,
                a number between [0, 0.5].
            seed: random seed.

        Returns:
            jittered_box: jittered box.
        """
        rand_numbers = tf.random_uniform(
            [1, 1, 4], minval=-ratio, maxval=ratio, dtype=tf.float32, seed=seed)
        box_width = tf.subtract(box[0, 0, 3], box[0, 0, 1])
        box_height = tf.subtract(box[0, 0, 2], box[0, 0, 0])
        hw_coefs = tf.stack([box_height, box_width, box_height, box_width])
        hw_rand_coefs = tf.multiply(hw_coefs, rand_numbers)
        jittered_box = tf.add(box, hw_rand_coefs)
        jittered_box = tf.clip_by_value(jittered_box, 0.0, 1.0)
        return jittered_box

    with tf.name_scope('RandomJitterBoxes', values=[boxes]):
        # boxes are [N, 4]. Lets first make them [N, 1, 1, 4]
        boxes_shape = tf.shape(boxes)
        boxes = tf.expand_dims(boxes, 1)
        boxes = tf.expand_dims(boxes, 2)

        distorted_boxes = tf.map_fn(
            lambda x: random_jitter_box(x, ratio, seed), boxes, dtype=tf.float32)

        distorted_boxes = tf.reshape(distorted_boxes, boxes_shape)

        return distorted_boxes

# -----------------------------------------------------------------
# CROP
# -----------------------------------------------------------------


def _strict_random_crop_image(image,
                              boxes,
                              labels,
                              is_crowd,
                              difficult,
                              masks=None,
                              sem_seg=None,
                              min_object_covered=1.0,
                              aspect_ratio_range=(0.75, 1.33),
                              area_range=(0.1, 1.0),
                              overlap_thresh=0.3):
    """Performs random crop.

    Note: boxes will be clipped to the crop. Keypoint coordinates that are
    outside the crop will be set to NaN, which is consistent with the original
    keypoint encoding for non-existing keypoints. This function always crops
    the image and is supposed to be used by `random_crop_image` function which
    sometimes returns image unchanged.

    Args:
        image: rank 3 float32 tensor containing 1 image -> [height, width, channels]
            with pixel values varying between [0, 1].
        boxes: rank 2 float32 tensor containing the bounding boxes with shape
            [num_instances, 4].
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].
        labels: rank 1 int32 tensor containing the object classes.
        masks: (optional) rank 3 float32 tensor with shape
            [num_instances, height, width] containing instance masks. The masks
            are of the same height, width as the input `image`.
        min_object_covered: the cropped image must cover at least this fraction of
                            at least one of the input bounding boxes.
        aspect_ratio_range: allowed range for aspect ratio of cropped image.
        area_range: allowed range for area ratio between cropped image and the
                    original image.
        overlap_thresh: minimum overlap thresh with new cropped
                        image to keep the box.

    Returns:
        image: image which is the same rank as input image.
        boxes: boxes which is the same rank as input boxes.
            Boxes are in normalized form.
        labels: new labels.

        If masks is not None, the function also returns:
        masks: rank 3 float32 tensor with shape [num_instances, height, width]
            containing instance masks.
    """
    with tf.name_scope('RandomCropImage', values=[image, boxes]):
        image_shape = tf.shape(image)

        # boxes are [N, 4]. Lets first make them [1, N, 4].
        boxes_expanded = tf.expand_dims(
            tf.clip_by_value(
                boxes, clip_value_min=0.0, clip_value_max=1.0), 0)

        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            image_shape,
            bounding_boxes=boxes_expanded,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)

        im_box_begin, im_box_size, im_box = sample_distorted_bounding_box

        new_image = tf.slice(image, im_box_begin, im_box_size)
        new_image.set_shape([None, None, image.get_shape()[2]])

        # [1, 4]
        im_box_rank2 = tf.squeeze(im_box, squeeze_dims=[0])
        # [4]
        im_box_rank1 = tf.squeeze(im_box)

        boxlist = box_list.BoxList(boxes)
        boxlist.add_field('labels', labels)
        boxlist.add_field('is_crowd', is_crowd)
        boxlist.add_field('difficult', difficult)
        if masks is not None:
            boxlist.add_field('masks', masks)

        im_boxlist = box_list.BoxList(im_box_rank2)

        # remove boxes that are outside cropped image
        boxlist, inside_window_ids = box_list_ops.prune_completely_outside_window(
            boxlist, im_box_rank1)

        # remove boxes that are outside image
        overlapping_boxlist, keep_ids = box_list_ops.prune_non_overlapping_boxes(
            boxlist, im_boxlist, overlap_thresh)

        # change the coordinate of the remaining boxes
        new_boxlist = box_list_ops.change_coordinate_frame(overlapping_boxlist,
                                                           im_box_rank1)
        new_boxes = new_boxlist.boxes
        new_boxes = tf.clip_by_value(
            new_boxes, clip_value_min=0.0, clip_value_max=1.0)
        new_boxes.set_shape([None, 4])
        result = [
            new_image,
            new_boxes,
            overlapping_boxlist.get_field('labels'),
            overlapping_boxlist.get_field('is_crowd'),
            overlapping_boxlist.get_field('difficult'),
        ]

        if masks is not None:
            masks_of_boxes_inside_window = tf.gather(masks, inside_window_ids)
            masks_of_boxes_completely_inside_window = tf.gather(
                masks_of_boxes_inside_window, keep_ids)
            masks_box_begin = [0, im_box_begin[0], im_box_begin[1]]
            masks_box_size = [-1, im_box_size[0], im_box_size[1]]
            new_masks = tf.slice(
                masks_of_boxes_completely_inside_window,
                masks_box_begin, masks_box_size)
            result.append(new_masks)
        if sem_seg is not None:
            sem_seg = tf.expand_dims(sem_seg, axis=-1)
            new_sem_seg = tf.slice(sem_seg, im_box_begin, im_box_size)
            new_sem_seg = tf.squeeze(new_sem_seg, axis=-1)
            new_sem_seg.set_shape([None, None])
            result.append(new_sem_seg)
        return tuple(result)


def random_crop_image(image,
                      boxes,
                      labels,
                      is_crowd,
                      difficult,
                      masks=None,
                      sem_seg=None,
                      min_object_covered=1.0,
                      aspect_ratio_range=(0.75, 1.33),
                      area_range=(0.1, 1.0),
                      overlap_thresh=0.3,
                      random_coef=0.0,
                      seed=None):
    """Randomly crops the image.

    Given the input image and its bounding boxes, this op randomly
    crops a subimage.  Given a user-provided set of input constraints,
    the crop window is resampled until it satisfies these constraints.
    If within 100 trials it is unable to find a valid crop, the original
    image is returned. See the Args section for a description of the input
    constraints. Both input boxes and returned Boxes are in normalized
    form (e.g., lie in the unit square [0, 1]).
    This function will return the original image with probability random_coef.

    Note: boxes will be clipped to the crop. Keypoint coordinates that are
    outside the crop will be set to NaN, which is consistent with the original
    keypoint encoding for non-existing keypoints.

    Args:
        image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
            with pixel values varying between [0, 1].
        boxes: rank 2 float32 tensor containing the bounding boxes with shape
            [num_instances, 4].
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].
        labels: rank 1 int32 tensor containing the object classes.
        label_scores: (optional) float32 tensor of shape [num_instances].
        representing the score for each box.
        masks: (optional) rank 3 float32 tensor with shape
            [num_instances, height, width] containing instance masks. The masks
            are of the same height, width as the input `image`.
        keypoints: (optional) rank 3 float32 tensor with shape
                [num_instances, num_keypoints, 2]. The keypoints are in y-x
                normalized coordinates.
        min_object_covered: the cropped image must cover at least this fraction of
                            at least one of the input bounding boxes.
        aspect_ratio_range: allowed range for aspect ratio of cropped image.
        area_range: allowed range for area ratio between cropped image and the
                    original image.
        overlap_thresh: minimum overlap thresh with new cropped
                        image to keep the box.
        random_coef: a random coefficient that defines the chance of getting the
                    original image. If random_coef is 0, we will always get the
                    cropped image, and if it is 1.0, we will always get the
                    original image.
        seed: random seed.

    Returns:
        image: Image shape will be [new_height, new_width, channels].
        boxes: boxes which is the same rank as input boxes. Boxes are in normalized
            form.
        labels: new labels.

        If label_scores, masks, or keypoints are not None, the function also
        returns:
        label_scores: new scores.
        masks: rank 3 float32 tensor with shape [num_instances, height, width]
            containing instance masks.
        keypoints: rank 3 float32 tensor with shape
                [num_instances, num_keypoints, 2]
    """

    def strict_random_crop_image_fn():
        return _strict_random_crop_image(
            image,
            boxes,
            labels,
            is_crowd,
            difficult,
            masks=masks,
            sem_seg=sem_seg,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            overlap_thresh=overlap_thresh)

    # avoids tf.cond to make faster RCNN training on borg. See b/140057645.
    if random_coef < sys.float_info.min:
        result = strict_random_crop_image_fn()
    else:
        do_a_crop_random = tf.random_uniform([], seed=seed)
        do_a_crop_random = tf.greater(do_a_crop_random, random_coef)

        outputs = [image, boxes, labels, is_crowd, difficult]

        if masks is not None:
            outputs.append(masks)
        if sem_seg is not None:
            outputs.append(sem_seg)

        result = tf.cond(
            do_a_crop_random,
            strict_random_crop_image_fn,
            lambda: tuple(outputs)
        )
    return result

# ----------------------------------------------------------------
#  main
# ----------------------------------------------------------------


def get_func_arg_map(
    load_instance_masks=False,
    load_semantic_mask=False
):
    """Returns the default mapping from a preprocessor function to its args.

    Args:
        load_instance_masks: If True, preprocessing functions will modify the
        instance masks

    Returns:
        A map from preprocessing functions to the arguments they receive.
    """

    func_arg_map = {
        random_horizontal_flip: [
            fields.InputFields.image,
            fields.InputFields.gt_boxes
        ],
        random_vertical_flip: [
            fields.InputFields.image,
            fields.InputFields.gt_boxes
        ],
        random_rotation90: [
            fields.InputFields.image,
            fields.InputFields.gt_boxes
        ],
        random_rotation90_both_direction:  [
            fields.InputFields.image,
            fields.InputFields.gt_boxes
        ],
        random_crop_image: [
            fields.InputFields.image,
            fields.InputFields.gt_boxes,
            fields.InputFields.gt_classes,
            fields.InputFields.gt_is_crowd,
            fields.InputFields.gt_difficult,
        ]
    }

    for func in func_arg_map:
        func_arg_map[func].append(
            fields.InputFields.gt_masks if load_instance_masks else None)
    for func in func_arg_map:
        func_arg_map[func].append(
            fields.InputFields.sem_seg if load_semantic_mask else None)
    func_arg_map.update(
        {
            random_pixel_value_scale: (fields.InputFields.image,),
            random_adjust_brightness: (fields.InputFields.image,),
            random_adjust_contrast: (fields.InputFields.image,),
            random_adjust_hue: (fields.InputFields.image,),
            random_adjust_saturation: (fields.InputFields.image,),
            random_distort_color: (fields.InputFields.image,),
            random_jitter_boxes: (fields.InputFields.gt_boxes,),
        }
    )
    return func_arg_map


def augment(
    cfg,
    tensor_dict,
    load_instance_masks,
    load_semantic_mask
):
    """Preprocess images and bounding boxes.

    Various types of preprocessing (to be implemented) based on the
    preprocess_options dictionary e.g. "crop image" (affects image and possibly
    boxes), "white balance image" (affects only image), etc. If self._options
    is None, no preprocessing is done.

    Args:
        cfg:
        tensor_dict: dictionary that contains images, boxes, and can contain other
                    things as well.
                    images-> rank 4 float32 tensor contains
                            1 image -> [1, height, width, 3].
                            with pixel values varying between [0, 1]
                    boxes-> rank 2 float32 tensor containing
                            the bounding boxes -> [N, 4].
                            Boxes are in normalized form meaning
                            their coordinates vary between [0, 1].
                            Each row is in the form
                            of [ymin, xmin, ymax, xmax].

    Returns:
        tensor_dict: which contains the preprocessed images, bounding boxes, etc.

    Raises:
        ValueError: (a) If the functions passed to Preprocess
                        are not in func_arg_map.
                    (b) If the arguments that a function needs
                        do not exist in tensor_dict.
                    (c) If image in tensor_dict is not rank 4
    """
    func_arg_map = get_func_arg_map(
        load_instance_masks, load_semantic_mask
    )
    # changes the images to image (rank 4 to rank 3) since the functions
    # receive rank 3 tensor for image
    if fields.InputFields.image in tensor_dict:
        images = tensor_dict[fields.InputFields.image]
        if len(images.get_shape()) != 3:
            raise ValueError('images in tensor_dict should be rank 3')

    preprocess_options = []
    if cfg.AUGMENT.HORIZONTAL_FLIP:
        preprocess_options.append((random_horizontal_flip, {}))
    if cfg.AUGMENT.VERTICAL_FLIP:
        preprocess_options.append((random_vertical_flip, {}))
    if cfg.AUGMENT.ROTATE:
        preprocess_options.append((random_rotation90, {}))
    if cfg.AUGMENT.ROTATE_BOTH_DIRECTION:
        preprocess_options.append((random_rotation90_both_direction, {}))
    if cfg.AUGMENT.PIXEL_VALUE_SCALE.ENABLED:
        preprocess_options.append(
            (random_pixel_value_scale,
             {"minval": cfg.AUGMENT.PIXEL_VALUE_SCALE.MIN_VALUE,
              "maxval": cfg.AUGMENT.PIXEL_VALUE_SCALE.MAX_VALUE}))
    if cfg.AUGMENT.ADJUST_BRIGHTNESS.ENABLED:
        preprocess_options.append(
            (random_adjust_brightness,
             {"max_delta": cfg.AUGMENT.ADJUST_BRIGHTNESS.MAX_DELTA}))
    if cfg.AUGMENT.ADJUST_CONSTRACT.ENABLED:
        preprocess_options.append(
            (random_adjust_contrast,
             {"min_delta": cfg.AUGMENT.ADJUST_CONSTRACT.MIN_DELTA,
              "max_delta": cfg.AUGMENT.ADJUST_CONSTRACT.MAX_DELTA}))
    if cfg.AUGMENT.ADJUST_HUE.ENABLED:
        preprocess_options.append(
            (random_adjust_hue,
             {"max_delta": cfg.AUGMENT.ADJUST_HUE.MAX_DELTA}))
    if cfg.AUGMENT.ADJUST_SATURATION.ENABLED:
        preprocess_options.append(
            (random_adjust_saturation,
             {"min_delta": cfg.AUGMENT.ADJUST_SATURATION.MIN_DELTA,
              "max_delta": cfg.AUGMENT.ADJUST_SATURATION.MAX_DELTA}))
    if cfg.AUGMENT.DISTORT_COLOR.ENABLED:
        preprocess_options.append(
            (random_distort_color,
             {"color_ordering": cfg.AUGMENT.DISTORT_COLOR.COLOR_ORDERING}))
    if cfg.AUGMENT.CROP.ENABLED:
        preprocess_options.append(
            (random_crop_image,
             {"min_object_covered": cfg.AUGMENT.CROP.MIN_OBJECT_COVERED,
              "aspect_ratio_range": cfg.AUGMENT.CROP.ASPECT_RATIO_RANGE,
              "area_range": cfg.AUGMENT.CROP.AREA_RANGE,
              "overlap_thresh": cfg.AUGMENT.CROP.OVERLAP_THRESH,
              "random_coef": cfg.AUGMENT.CROP.RANDOM_COEF}))
    if cfg.AUGMENT.JITTER_BOX.ENABLED:
        preprocess_options.append(
            (random_jitter_boxes, {"ratio": cfg.AUGMENT.JITTER_BOX.RATIO}))

    # Preprocess inputs based on preprocess_options
    for option in preprocess_options:
        func, params = option
        if func not in func_arg_map:
            raise ValueError('The function %s does not exist in func_arg_map' %
                             (func.__name__))
        arg_names = func_arg_map[func]
        for a in arg_names:
            if a not in tensor_dict and a is not None:
                raise ValueError('The function %s requires argument %s' %
                                 (func.__name__, a))
        args = [tensor_dict.get(a, None) for a in arg_names]
        results = func(*args, **params)
        if not isinstance(results, (list, tuple)):
            results = (results,)
        # Removes None args since the return values will not contain those.
        arg_names = [arg_name for arg_name in arg_names if arg_name is not None]
        for res, arg_name in zip(results, arg_names):
            tensor_dict[arg_name] = res

    return tensor_dict


def compute_new_shape(orig_shape, min_dimension, max_dimension):
    """Compute new shape for resize_image method."""
    orig_shape = tf.cast(orig_shape, tf.float32)
    orig_min_dim = tf.reduce_min(orig_shape)
    # Calculates the larger of the possible sizes
    min_dimension = tf.cast(min_dimension, tf.float32)
    large_scale_factor = min_dimension / orig_min_dim
    # Scaling orig_(height|width) by large_scale_factor will make the smaller
    # dimension equal to min_dimension, save for floating point rounding errors.
    # For reasonably-sized images, taking the nearest integer will reliably
    # eliminate this error.
    large_shape = tf.to_int32(tf.round(orig_shape * large_scale_factor))
    if max_dimension:
        # Calculates the smaller of the possible sizes, use that if the larger
        # is too big.
        orig_max_dim = tf.reduce_max(orig_shape)
        max_dimension = tf.cast(max_dimension, tf.float32)
        small_scale_factor = max_dimension / orig_max_dim
        # Scaling orig_(height|width) by small_scale_factor will make the larger
        # dimension equal to max_dimension, save for floating point rounding
        # errors. For reasonably-sized images, taking the nearest integer will
        # reliably eliminate this error.
        small_shape = tf.to_int32(tf.round(orig_shape * small_scale_factor))
        new_shape = tf.cond(
            tf.cast(tf.reduce_max(large_shape), tf.float32) > max_dimension,
            lambda: small_shape, lambda: large_shape)
    else:
        new_shape = large_shape
    return new_shape


def resize(cfg, tensor_dict, training=False):
    """resize images.
    Args:

    Raises:

    Returns:

    """
    input_fields = fields.InputFields
    if training:
        min_dimension = random.choice(cfg.TRANSFORM.RESIZE.MIN_SIZE_TRAIN)
        max_dimension = cfg.TRANSFORM.RESIZE.MAX_SIZE_TRAIN
    else:
        min_dimension = cfg.TRANSFORM.RESIZE.MIN_SIZE_TEST
        max_dimension = cfg.TRANSFORM.RESIZE.MAX_SIZE_TEST
    use_mini_masks = cfg.TRANSFORM.RESIZE.USE_MINI_MASKS
    mini_mask_size = cfg.TRANSFORM.RESIZE.MINI_MASK_SIZE

    if input_fields.image not in tensor_dict:
        raise ValueError("no image in tensor_dict.")
    image = tensor_dict[input_fields.image]
    boxes = tensor_dict.get(input_fields.gt_boxes)
    masks = tensor_dict.get(input_fields.gt_masks)
    sem_seg = tensor_dict.get(input_fields.sem_seg)

    if len(image.get_shape()) != 3:
        raise ValueError('Image should be 3D tensor')
    with tf.name_scope('ResizeImage', values=[image]):
        orig_shape = tf.cast(tf.shape(image)[:2], tf.float32)
        new_shape = compute_new_shape(orig_shape, min_dimension, max_dimension)

        new_image = resize_images(image, new_shape, align_corners=True)
        tensor_dict[input_fields.image] = new_image

        new_sem_seg = None
        if sem_seg is not None:
            sem_seg = tf.expand_dims(sem_seg, axis=2)
            new_sem_seg = resize_images(
                sem_seg, new_shape, method="nearest", align_corners=True
            )
            new_sem_seg = tf.squeeze(new_sem_seg, axis=2)
            tensor_dict[input_fields.sem_seg] = new_sem_seg

        new_masks = None
        if masks is not None:
            masks = tf.expand_dims(masks, axis=3)
            if use_mini_masks:
                if boxes is None:
                    raise ValueError('`boxes` must be given.')
                box_ind = tf.range(tf.shape(masks)[0])
                mini_mask_shape = [mini_mask_size, mini_mask_size]
                new_masks = tf.image.crop_and_resize(
                    masks,
                    boxes,
                    box_ind,
                    mini_mask_shape,
                    method='bilinear'
                )
            else:
                new_masks = resize_images(
                    masks, new_shape, method="bilinear", align_corners=True
                )
            new_masks = tf.squeeze(new_masks, axis=3)
            new_masks = tf.round(new_masks)
            tensor_dict[input_fields.gt_masks] = new_masks

        tensor_dict[input_fields.true_shape] = new_shape
    return tensor_dict


def run(
    cfg,
    tensor_dict,
    training=False,
    load_instance_masks=False,
    load_semantic_mask=False
):
    input_fields = fields.InputFields
    if training:
        tensor_dict = augment(
            cfg,
            tensor_dict,
            load_instance_masks=load_instance_masks,
            load_semantic_mask=load_semantic_mask,
        )

    tensor_dict = resize(cfg, tensor_dict, training)
    if input_fields.gt_boxes in tensor_dict:
        boxes = tensor_dict[input_fields.gt_boxes]
        new_shape = tensor_dict[input_fields.true_shape]

        boxlist = box_list.BoxList(boxes)
        absolute_boxlist = box_list_ops.to_absolute_coordinates(boxlist, new_shape)
        tensor_dict[input_fields.gt_boxes] = absolute_boxlist.boxes

        is_valid = tf.ones([tf.shape(boxes)[0]], dtype=tf.bool)
        tensor_dict[input_fields.is_valid] = is_valid
    return tensor_dict
