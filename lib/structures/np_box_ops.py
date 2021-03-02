"""Operations for [N, 4] numpy arrays representing bounding boxes.

Example box operations that are supported:
  * Areas: compute bounding box areas
  * IOU: pairwise intersection-over-union scores
"""
import numpy as np


def area(boxes):
    """Computes area of boxes.

    Args:
        boxes: Numpy array with shape [N, 4] holding N boxes

    Returns:
        a numpy array with shape [N*1] representing box areas
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes
        boxes2: a numpy array with shape [M, 4] holding M boxes

    Returns:
        a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape),
        all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape),
        all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def iou(boxes1, boxes2):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes.
        boxes2: a numpy array with shape [M, 4] holding N boxes.

    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2)
    area1 = area(boxes1)
    area2 = area(boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(
        area2, axis=0) - intersect
    return intersect / union


def ioa(boxes1, boxes2):
    """Computes pairwise intersection-over-area between box collections.

    Intersection-over-area (ioa) between two boxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, IOA(box1, box2) != IOA(box2, box1).

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes.
        boxes2: a numpy array with shape [M, 4] holding N boxes.

    Returns:
        a numpy array with shape [N, M] representing pairwise ioa scores.
    """
    intersect = intersection(boxes1, boxes2)
    areas = np.expand_dims(area(boxes2), axis=0)
    return intersect / areas


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.

    Args:
        boxes: [N, (y1, x1, y2, x2)] boxes to update.
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply.

    Return:
        Refined boxes.
    """
    # Convert to y, x, h, w
    ymin, xmin, ymax, xmax = np.split(boxes, 4, axis=1)
    height = ymax - ymin
    width = xmax - xmin
    center_y = ymin + 0.5 * height
    center_x = xmin + 0.5 * width
    # Apply deltas
    dy, dx, dh, dw = np.split(deltas, 4, axis=1)
    center_y += dy * height
    center_x += dx * width
    height *= np.exp(dh)
    width *= np.exp(dw)
    # Convert back to y1, x1, y2, x2
    ymin = center_y - 0.5 * height
    xmin = center_x - 0.5 * width
    ymax = ymin + height
    xmax = xmin + width
    result = np.concatenate([ymin, xmin, ymax, xmax], axis=1)
    return result
