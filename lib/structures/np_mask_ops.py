"""Operations for [N, height, width] numpy arrays representing masks.
Notice the masks cover the box regions instead of the whole images when boxes provided.

Example mask operations that are supported:
  * Areas: compute mask areas
  * IOU: pairwise intersection-over-union scores
"""
import numpy as np
import cv2

from . import np_box_ops

EPSILON = 1e-7


def area(masks, boxes=None):
    """Computes area of masks.

    Args:
      masks: Numpy array with shape [N, height, width] holding N masks. Masks
        values are of type np.uint8 and values are in {0,1}.
      boxes: Numpy array with shape [N, 4] holding N boxes

    Returns:
      a numpy array with shape [N*1] representing mask areas.

    Raises:
      ValueError: If masks.dtype is not np.uint8
    """
    if masks.dtype != np.uint8:
        raise ValueError('Masks type should be np.uint8')
    if boxes is not None:
        box_area = np_box_ops.area(boxes)
        mask_ratio = np.mean(masks, axis=(1, 2), dtype=np.float32)
        return box_area * mask_ratio
    return np.sum(masks, axis=(1, 2), dtype=np.float32)


def intersection(masks1, masks2, boxes1=None, boxes2=None):
    """Compute pairwise intersection areas between masks.

    Args:
      masks1: a numpy array with shape [N, height, width] holding N masks. Masks
        values are of type np.uint8 and values are in {0,1}.
      masks2: a numpy array with shape [M, height, width] holding M masks. Masks
        values are of type np.uint8 and values are in {0,1}.
      boxes1: a numpy array with shape [N, 4] holding N boxes.
      boxes2: a numpy array with shape [M, 4] holding M boxes.

    Returns:
      a numpy array with shape [N*M] representing pairwise intersection area.

    Raises:
      ValueError: If masks1 and masks2 are not of type np.uint8.
    """
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError('masks1 and masks2 should be of type np.uint8')
    n = masks1.shape[0]
    m = masks2.shape[0]
    answer = np.zeros([n, m], dtype=np.float32)
    if boxes1 is None or boxes2 is None:
        for i in np.arange(n):
            for j in np.arange(m):
                answer[i, j] = np.sum(np.minimum(masks1[i], masks2[j]), dtype=np.float32)
    else:
        masks1 = [get_resized_box_mask(masks1[i], boxes1[i]) for i in range(n)]
        masks2 = [get_resized_box_mask(masks2[j], boxes2[j]) for j in range(m)]
        [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
        [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

        all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
        all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
        intersect_heights = all_pairs_min_ymax - all_pairs_max_ymin
        all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
        all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
        intersect_widths = all_pairs_min_xmax - all_pairs_max_xmin
        for i in np.arange(n):
            for j in np.arange(m):
                height = intersect_heights[i, j]
                width = intersect_widths[i, j]
                if height <= 0 or width <= 0:
                    continue
                answer[i, j] = np.sum(
                    np.minimum(
                        masks1[i][
                            (y_max1[i] - height):(y_max1[i] + 1),
                            (x_max1[i] - width):(x_max1[i] + 1)
                        ], 
                        masks2[j][
                            (y_max2[i] - height):(y_max2[i] + 1),
                            (x_max2[i] - width):(x_max2[i] + 1)
                        ]
                    ), 
                    dtype=np.float32
                )
    return answer


def iou(masks1, masks2, boxes1=None, boxes2=None):
    """Computes pairwise intersection-over-union between mask collections.

    Args:
        masks1: a numpy array with shape [N, height, width] holding N masks. Masks
            values are of type np.uint8 and values are in {0,1}.
        masks2: a numpy array with shape [M, height, width] holding N masks. Masks
            values are of type np.uint8 and values are in {0,1}.
        boxes1: a numpy array with shape [N, 4] holding N boxes.
        boxes2: a numpy array with shape [M, 4] holding M boxes.

    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.

    Raises:
        ValueError: If masks1 and masks2 are not of type np.uint8.
    """
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError('masks1 and masks2 should be of type np.uint8')
    intersect = intersection(masks1, masks2, boxes1, boxes2)
    area1 = area(masks1, boxes1)
    area2 = area(masks2, boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect
    return intersect / np.maximum(union, EPSILON)


def ioa(masks1, masks2, boxes1=None, boxes2=None):
    """Computes pairwise intersection-over-area between box collections.

    Intersection-over-area (ioa) between two masks, mask1 and mask2 is defined as
    their intersection area over mask2's area. Note that ioa is not symmetric,
    that is, IOA(mask1, mask2) != IOA(mask2, mask1).

    Args:
        masks1: a numpy array with shape [N, height, width] holding N masks. Masks
            values are of type np.uint8 and values are in {0,1}.
        masks2: a numpy array with shape [M, height, width] holding N masks. Masks
            values are of type np.uint8 and values are in {0,1}.
        boxes1: a numpy array with shape [N, 4] holding N boxes.
        boxes2: a numpy array with shape [M, 4] holding M boxes.

    Returns:
        a numpy array with shape [N, M] representing pairwise ioa scores.

    Raises:
        ValueError: If masks1 and masks2 are not of type np.uint8.
    """
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError('masks1 and masks2 should be of type np.uint8')
    intersect = intersection(masks1, masks2, boxes1, boxes2)
    areas = np.expand_dims(area(masks2, boxes2), axis=0)
    return intersect / (areas + EPSILON)


def paste_masks_into_image(masks, boxes, image_shape, mask_threshold=0.5):
    """Paste the box masks back into full image masks.
    Embeds masks in bounding boxes of larger masks whose shapes correspond to
    image shape.
    Args:
        masks: [N, mask_height, mask_width], the box masks.
        boxes: [N, 4], the box coordinates ymin, xmin, ymax, xmax.
        image_shape: The output mask will have the same shape as the image shape.
        mask_threshold: 
    Returns:
        masks: [N, image_height, image_width], the image masks.
    """
    n = masks.shape[0]
    image_masks = np.zeros([n, *image_shape], dtype=np.uint8)
    boxes = np.round(boxes).astype(np.int32)
    for i in range(n):
        box = boxes[i]
        y_0 = max(box[0], 0)
        y_1 = min(box[2] + 1, image_shape[0])
        x_0 = max(box[1], 0)
        x_1 = min(box[3] + 1, image_shape[1])
        box_mask = get_resized_box_mask(masks[i], box, mask_threshold)
        image_masks[i][y_0:y_1, x_0:x_1] = box_mask[:y_1-y_0, :x_1-x_0]
    return image_masks


def get_resized_box_mask(mask, box, mask_threshold=0.5):
    """Same as `paste_masks_into_image`."""
    samples_h = box[2] - box[0] + 1
    samples_w = box[3] - box[1] + 1

    mask = cv2.resize(mask, (samples_w, samples_h), interpolation=cv2.INTER_LINEAR)
    if mask_threshold > 0:
        mask = (mask > mask_threshold).astype(np.uint8)
    else:
        mask = np.round(mask).astype(np.uint8)
    return mask