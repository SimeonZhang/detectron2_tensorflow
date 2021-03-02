"""Bounding Box List operations.

Example box operations that are supported:
  * areas: compute bounding box areas
  * iou: pairwise intersection-over-union scores
  * sq_dist: pairwise distances between bounding boxes

Whenever box_list_ops functions output a BoxList, the fields of the incoming
BoxList are retained unless documented otherwise.
"""
from enum import Enum, unique
import tensorflow as tf
import math

from .box_list import BoxList
from ..utils import shape_utils


@unique
class SortOrder(Enum):
    """Enum class for sort order.

    Attributes:
      ascend: ascend order.
      descend: descend order.
    """
    ASCENDING = 1
    DESCENDING = 2


def area(boxlist, scope=None):
    """Computes area of boxes.

    Args:
        boxlist: BoxList holding N boxes
        scope: name scope.

    Returns:
        a tensor with shape [N] representing box areas.
    """
    with tf.name_scope(scope, 'Area'):
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.boxes, num_or_size_splits=4, axis=1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def center(boxlist, scope=None):
    """Computes center of boxes.

    Args:
        boxlist: BoxList holding N boxes
        scope: name scope.

    Returns:
        a tensor with shape [N, 2] representing box centers.
    """
    with tf.name_scope(scope, 'Center'):
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.boxes, num_or_size_splits=4, axis=1)
        return (
            tf.squeeze((y_min + x_min) / 2., [1]),
            tf.squeeze((x_min + x_max) / 2., [1])
        )


def height_width(boxlist, scope=None):
    """Computes height and width of boxes in boxlist.

    Args:
        boxlist: BoxList holding N boxes
        scope: name scope.

    Returns:
        Height: A tensor with shape [N] representing box heights.
        Width: A tensor with shape [N] representing box widths.
    """
    with tf.name_scope(scope, 'HeightWidth'):
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.boxes, num_or_size_splits=4, axis=1)
        return (
            tf.squeeze(y_max - y_min, [1]),
            tf.squeeze(x_max - x_min, [1])
        )


def scale(boxlist, y_scale, x_scale, scope=None):
    """scale box coordinates in x and y dimensions.

    Args:
      boxlist: BoxList holding N boxes
      y_scale: (float) scalar tensor
      x_scale: (float) scalar tensor
      scope: name scope.

    Returns:
      boxlist: BoxList holding N boxes
    """
    with tf.name_scope(scope, 'Scale'):
        y_scale = tf.cast(y_scale, tf.float32)
        x_scale = tf.cast(x_scale, tf.float32)
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.boxes, num_or_size_splits=4, axis=1)
        y_min = y_scale * y_min
        y_max = y_scale * y_max
        x_min = x_scale * x_min
        x_max = x_scale * x_max
        scaled_boxlist = BoxList(
            tf.concat([y_min, x_min, y_max, x_max], 1))
        return _copy_extra_datas(scaled_boxlist, boxlist)


def clip_to_window(boxlist, window, filter_nonoverlapping=True, scope=None):
    """Clip bounding boxes to a window.

    This op clips any input bounding boxes (represented by bounding box
    corners) to a window, optionally filtering out boxes that do not
    overlap at all with the window.

    Args:
      boxlist: BoxList holding M_in boxes
      window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
        window to which the op should clip boxes.
      filter_nonoverlapping: whether to filter out boxes that do not overlap at
        all with the window.
      scope: name scope.

    Returns:
      a BoxList holding M_out boxes where M_out <= M_in
    """
    with tf.name_scope(scope, 'ClipToWindow'):
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.boxes, num_or_size_splits=4, axis=1)
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
        y_min_clipped = tf.maximum(tf.minimum(y_min, win_y_max), win_y_min)
        y_max_clipped = tf.maximum(tf.minimum(y_max, win_y_max), win_y_min)
        x_min_clipped = tf.maximum(tf.minimum(x_min, win_x_max), win_x_min)
        x_max_clipped = tf.maximum(tf.minimum(x_max, win_x_max), win_x_min)
        clipped = BoxList(
            tf.concat([y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped],
                      1))
        clipped = _copy_extra_datas(clipped, boxlist)
        if filter_nonoverlapping:
            areas = area(clipped)
            nonzero_area_indices = tf.cast(
                tf.reshape(tf.where(tf.greater(areas, 0.0)), [-1]), tf.int32)
            clipped = gather(clipped, nonzero_area_indices)
        return clipped


def inside_window(boxlist, window, boundary_threshold=0., scope=None):
    with tf.name_scope(scope, 'InsideWindow'):
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.boxes, num_or_size_splits=4, axis=1)
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
        coordinate_violations = tf.concat([
            tf.less(y_min, win_y_min - boundary_threshold),
            tf.less(x_min, win_x_min - boundary_threshold),
            tf.greater(y_max, win_y_max + boundary_threshold),
            tf.greater(x_max, win_x_max + boundary_threshold)
        ], 1)
        return tf.logical_not(tf.reduce_any(coordinate_violations, 1))


def prune_outside_window(boxlist, window, scope=None):
    """Prunes bounding boxes that fall outside a given window.

    This function prunes bounding boxes that even partially fall outside the given
    window. See also clip_to_window which only prunes bounding boxes that fall
    completely outside the window, and clips any bounding boxes that partially
    overflow.

    Args:
        boxlist: a BoxList holding M_in boxes.
        window: a float tensor of shape [4] representing [ymin, xmin, ymax, xmax]
          of the window
        scope: name scope.

    Returns:
        pruned_corners: a tensor with shape [M_out, 4] where M_out <= M_in
        valid_indices: a tensor with shape [M_out] indexing the valid bounding boxes
          in the input tensor.
    """
    with tf.name_scope(scope, 'PruneOutsideWindow'):
        cond_inside = inside_window(boxlist, window)
        valid_indices = tf.reshape(tf.where(cond_inside), [-1])
        return gather(boxlist, valid_indices), valid_indices


def prune_completely_outside_window(boxlist, window, scope=None):
    """Prunes bounding boxes that fall completely outside of the given window.

    The function clip_to_window prunes bounding boxes that fall
    completely outside the window, but also clips any bounding boxes that
    partially overflow. This function does not clip partially overflowing boxes.

    Args:
        boxlist: a BoxList holding M_in boxes.
        window: a float tensor of shape [4] representing [ymin, xmin, ymax, xmax]
          of the window
        scope: name scope.

    Returns:
        pruned_corners: a tensor with shape [M_out, 4] where M_out <= M_in
        valid_indices: a tensor with shape [M_out] indexing the valid bounding boxes
          in the input tensor.
    """
    with tf.name_scope(scope, 'PruneCompleteleyOutsideWindow'):
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.boxes, num_or_size_splits=4, axis=1)
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
        coordinate_violations = tf.concat([
            tf.greater_equal(y_min, win_y_max), tf.greater_equal(x_min, win_x_max),
            tf.less_equal(y_max, win_y_min), tf.less_equal(x_max, win_x_min)
        ], 1)
        valid_indices = tf.reshape(
            tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
        return gather(boxlist, valid_indices), valid_indices


def pairwise_intersection(boxlist1, boxlist2, scope=None):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxlist1: BoxList holding N boxes
        boxlist2: BoxList holding M boxes
        scope: name scope.

    Returns:
        a tensor with shape [N, M] representing pairwise intersections
    """
    with tf.name_scope(scope, 'PairwiseIntersection'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(
            value=boxlist1.boxes, num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(
            value=boxlist2.boxes, num_or_size_splits=4, axis=1)
        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths


def pairwise_convex(boxlist1, boxlist2, scope=None):
    """Compute pairwise convex areas between boxes.

    Args:
        boxlist1: BoxList holding N boxes
        boxlist2: BoxList holding M boxes
        scope: name scope.

    Returns:
        a tensor with shape [N, M] representing pairwise convexs
    """
    with tf.name_scope(scope, 'PairwiseConvex'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(
            value=boxlist1.boxes, num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(
            value=boxlist2.boxes, num_or_size_splits=4, axis=1)
        all_pairs_max_ymax = tf.maximum(y_max1, tf.transpose(y_max2))
        all_pairs_min_ymin = tf.minimum(y_min1, tf.transpose(y_min2))
        convex_heights = tf.maximum(0.0, all_pairs_max_ymax - all_pairs_min_ymin)
        all_pairs_max_xmax = tf.maximum(x_max1, tf.transpose(x_max2))
        all_pairs_min_xmin = tf.minimum(x_min1, tf.transpose(x_min2))
        convex_widths = tf.maximum(0.0, all_pairs_max_xmax - all_pairs_min_xmin)
        return convex_heights * convex_widths


def matched_intersection(boxlist1, boxlist2, scope=None):
    """Compute intersection areas between corresponding boxes in two boxlists.

    Args:
        boxlist1: BoxList holding N boxes
        boxlist2: BoxList holding N boxes
        scope: name scope.

    Returns:
        a tensor with shape [N] representing pairwise intersections
    """
    with tf.name_scope(scope, 'MatchedIntersection'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(
            value=boxlist1.boxes, num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(
            value=boxlist2.boxes, num_or_size_splits=4, axis=1)
        min_ymax = tf.minimum(y_max1, y_max2)
        max_ymin = tf.maximum(y_min1, y_min2)
        intersect_heights = tf.maximum(0.0, min_ymax - max_ymin)
        min_xmax = tf.minimum(x_max1, x_max2)
        max_xmin = tf.maximum(x_min1, x_min2)
        intersect_widths = tf.maximum(0.0, min_xmax - max_xmin)
        return tf.reshape(intersect_heights * intersect_widths, [-1])


def pairwise_iou(boxlist1, boxlist2, iou_type='iou', scope=None):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxlist1: BoxList holding N boxes
        boxlist2: BoxList holding M boxes
        scope: name scope.

    Returns:
        a tensor with shape [N, M] representing pairwise iou scores.
    """
    with tf.name_scope(scope, 'PairwiseIOU'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(
            value=boxlist1.boxes, num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(
            value=boxlist2.boxes, num_or_size_splits=4, axis=1)

        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)

        intersections = intersect_heights * intersect_widths

        height1 = y_max1 - y_min1
        width1 = x_max1 - x_min1
        areas1 = height1 * width1
        height2 = y_max2 - y_min2
        width2 = x_max2 - x_min2
        areas2 = height2 * width2

        unions = areas1 + tf.transpose(areas2) - intersections

        iou = tf.where(
            tf.equal(unions, 0.0),
            tf.zeros_like(unions), tf.truediv(intersections, unions)
        )

        if iou_type != "iou":
            all_pairs_max_ymax = tf.maximum(y_max1, tf.transpose(y_max2))
            all_pairs_min_ymin = tf.minimum(y_min1, tf.transpose(y_min2))
            convex_heights = tf.maximum(0.0, all_pairs_max_ymax - all_pairs_min_ymin)
            all_pairs_max_xmax = tf.maximum(x_max1, tf.transpose(x_max2))
            all_pairs_min_xmin = tf.minimum(x_min1, tf.transpose(x_min2))
            convex_widths = tf.maximum(0.0, all_pairs_max_xmax - all_pairs_min_xmin)

            if iou_type == "giou":
                convex = convex_heights * intersect_widths
                giou = iou - tf.where(
                    tf.equal(convex, 0.0),
                    tf.zeros_like(convex), tf.truediv(convex - unions, convex)
                )
                return giou

            convex_diagonal_squares = tf.add(
                tf.square(all_pairs_max_ymax - all_pairs_min_ymin),
                tf.square(all_pairs_max_xmax - all_pairs_min_xmin)
            )
            centerpoint_distance_squares = tf.add(
                tf.square((x_min1 + x_max1) / 2. - tf.transpose(x_min2 + x_max2) / 2.),
                tf.square((y_min1 + y_max1) / 2. - tf.transpose(y_min2 + y_max2) / 2.)
            )
            diou = iou - tf.where(
                tf.equal(convex_diagonal_squares, 0.0),
                tf.zeros_like(convex_diagonal_squares),
                tf.truediv(centerpoint_distance_squares, convex_diagonal_squares)
            )
            if iou_type == "diou":
                return diou
            if iou_type == "ciou":
                v = 4 / math.pi ** 2 * (
                    tf.atan(width1 / height1) - tf.transpose(tf.atan(width2 / height2))) ** 2
                alpha = v / (1. - iou + v)
                ciou = diou - alpha * v
                return ciou
        return iou


def matched_iou(boxlist1, boxlist2, iou_type="iou", scope=None):
    """Compute intersection-over-union between corresponding boxes in boxlists.

    Args:
        boxlist1: BoxList holding N boxes
        boxlist2: BoxList holding N boxes
        scope: name scope.

    Returns:
        a tensor with shape [N] representing pairwise iou scores.
    """
    with tf.name_scope(scope, 'MatchedIOU'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(
            value=boxlist1.boxes, num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(
            value=boxlist2.boxes, num_or_size_splits=4, axis=1)

        min_ymax = tf.minimum(y_max1, y_max2)
        max_ymin = tf.maximum(y_min1, y_min2)
        intersect_heights = tf.maximum(0.0, min_ymax - max_ymin)
        min_xmax = tf.minimum(x_max1, x_max2)
        max_xmin = tf.maximum(x_min1, x_min2)
        intersect_widths = tf.maximum(0.0, min_xmax - max_xmin)

        intersections = intersect_heights * intersect_widths

        height1 = y_max1 - y_min1
        width1 = x_max1 - x_min1
        areas1 = height1 * width1
        height2 = y_max2 - y_min2
        width2 = x_max1 - x_min2
        areas2 = height2 * width2

        unions = areas1 + areas2 - intersections

        iou = tf.where(
            tf.equal(unions, 0.0),
            tf.zeros_like(unions), tf.truediv(intersections, unions)
        )

        if iou_type != "iou":
            max_ymax = tf.maximum(y_max1, y_max2)
            min_ymin = tf.minimum(y_min1, y_min2)
            convex_heights = tf.maximum(0.0, max_ymax - min_ymin)
            max_xmax = tf.maximum(x_max1, x_max2)
            min_xmin = tf.minimum(x_min1, x_min2)
            convex_widths = tf.maximum(0.0, max_xmax - min_xmin)

            if iou_type == "giou":
                convex = convex_heights * intersect_widths
                giou = iou - tf.where(
                    tf.equal(convex, 0.0),
                    tf.zeros_like(convex), tf.truediv(convex - unions, convex)
                )
                return giou

            convex_diagonal_squares = tf.add(
                tf.square(max_ymax - min_ymin), tf.square(max_xmax - min_xmin)
            )
            centerpoint_distance_squares = tf.add(
                tf.square((x_min1 + x_max1) / 2. - (x_min2 + x_max2) / 2.),
                tf.square((y_min1 + y_max1) / 2. - (y_min2 + y_max2) / 2.)
            )
            diou = iou - tf.where(
                tf.equal(convex_diagonal_squares, 0.0),
                tf.zeros_like(convex_diagonal_squares),
                tf.truediv(centerpoint_distance_squares, convex_diagonal_squares)
            )
            if iou_type == "diou":
                return diou
            if iou_type == "ciou":
                v = 4 / math.pi ** 2 * (tf.atan(width1 / height1) - tf.atan(width2 / height2)) ** 2
                alpha = v / (1. - iou + v)
                ciou = diou - alpha * v
                return ciou
        return iou


def pairwise_ioa(boxlist1, boxlist2, scope=None):
    """Computes pairwise intersection-over-area between box collections.

    intersection-over-area (IOA) between two boxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, ioa(box1, box2) != ioa(box2, box1).

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding M boxes
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise ioa scores.
    """
    with tf.name_scope(scope, 'PairwiseIOA'):
        intersections = pairwise_intersection(boxlist1, boxlist2)
        areas = tf.expand_dims(area(boxlist2), 0)
        return tf.truediv(intersections, areas)


def prune_non_overlapping_boxes(
        boxlist1, boxlist2, min_overlap=0.0, scope=None):
    """Prunes the boxes in boxlist1 that overlap less than thresh with boxlist2.

    For each box in boxlist1, we want its IOA to be more than minoverlap with
    at least one of the boxes in boxlist2. If it does not, we remove it.

    Args:
        boxlist1: BoxList holding N boxes.
        boxlist2: BoxList holding M boxes.
        min_overlap: Minimum required overlap between boxes, to count them as
                    overlapping.
        scope: name scope.

    Returns:
        new_boxlist1: A pruned boxlist with size [N', 4].
        keep_inds: A tensor with shape [N'] indexing kept bounding boxes in the
          first input BoxList `boxlist1`.
    """
    with tf.name_scope(scope, 'PruneNonOverlappingBoxes'):
        ioa_ = pairwise_ioa(boxlist2, boxlist1)  # [M, N] tensor
        ioa_ = tf.reduce_max(ioa_, reduction_indices=[0])  # [N] tensor
        keep_bool = tf.greater_equal(ioa_, tf.constant(min_overlap))
        keep_inds = tf.squeeze(tf.where(keep_bool), squeeze_dims=[1])
        new_boxlist1 = gather(boxlist1, keep_inds)
        return new_boxlist1, keep_inds


def prune_small_boxes(boxlist, min_side, scope=None):
    """Prunes small boxes in the boxlist which have a side smaller than min_side.

    Args:
        boxlist: BoxList holding N boxes.
        min_side: Minimum width AND height of box to survive pruning.
        scope: name scope.

    Returns:
        A pruned boxlist.
    """
    with tf.name_scope(scope, 'PruneSmallBoxes'):
        height, width = height_width(boxlist)
        is_valid = tf.logical_and(tf.greater_equal(width, min_side),
                                  tf.greater_equal(height, min_side))
        return gather(boxlist, tf.reshape(tf.where(is_valid), [-1]))


def change_coordinate_frame(boxlist, window, scope=None):
    """Change coordinate frame of the boxlist to be relative to window's frame.

    Given a window of the form [ymin, xmin, ymax, xmax],
    changes bounding box coordinates from boxlist to be relative to this window
    (e.g., the min corner maps to (0,0) and the max corner maps to (1,1)).

    An example use case is data augmentation: where we are given groundtruth
    boxes (boxlist) and would like to randomly crop the image to some
    window (window). In this case we need to change the coordinate frame of
    each groundtruth box to be relative to this new window.

    Args:
        boxlist: A BoxList object holding N boxes.
        window: A rank 1 tensor [4].
        scope: name scope.

    Returns:
        Returns a BoxList object with N boxes.
    """
    with tf.name_scope(scope, 'ChangeCoordinateFrame'):
        win_height = window[2] - window[0]
        win_width = window[3] - window[1]
        boxlist_new = scale(
            BoxList(
                boxlist.boxes - [window[0], window[1], window[0], window[1]]),
            1.0 / win_height, 1.0 / win_width)
        boxlist_new = _copy_extra_datas(boxlist_new, boxlist)
        return boxlist_new


def boolean_mask(boxlist, indicator, fields=None, scope=None):
    """Select boxes from BoxList according to indicator and return new BoxList.

    `boolean_mask` returns the subset of boxes that are marked as "True" by the
    indicator tensor. By default, `boolean_mask` returns boxes corresponding to
    the input index list, as well as all additional fields stored in the boxlist
    (indexing into the first dimension).  However one can optionally only draw
    from a subset of fields.

    Args:
        boxlist: BoxList holding N boxes
        indicator: a rank-1 boolean tensor
        fields: (optional) list of fields to also gather from.  If None (default),
          all fields are gathered from.  Pass an empty fields list to only gather
          the box coordinates.
        scope: name scope.

    Returns:
        subboxlist: a BoxList corresponding to the subset of the input BoxList
            specified by indicator
    Raises:
        ValueError: if `indicator` is not a rank-1 boolean tensor.
    """
    with tf.name_scope(scope, 'BooleanMask'):
        if indicator.shape.ndims != 1:
            raise ValueError('indicator should have rank 1')
        if indicator.dtype != tf.bool:
            raise ValueError('indicator should be a boolean tensor')
        subboxlist = BoxList(tf.boolean_mask(boxlist.boxes, indicator))
        if fields is None:
            fields = boxlist.get_extra_fields()
        for field in fields:
            if not boxlist.has_field(field):
                raise ValueError('boxlist must contain all specified fields')
            subfieldlist = tf.boolean_mask(boxlist.get_field(field), indicator)
            subboxlist.add_field(field, subfieldlist)
        for tracking in boxlist.get_all_trackings():
            subboxlist.set_tracking(tracking, boxlist.get_tracking(tracking))
        return subboxlist


def gather(boxlist, indices, fields=None, scope=None):
    """Gather boxes from BoxList according to indices and return new BoxList.

    By default, `gather` returns boxes corresponding to the input index list, as
    well as all additional fields stored in the boxlist (indexing into the
    first dimension).  However one can optionally only gather from a
    subset of fields.

    Args:
      boxlist: BoxList holding N boxes
      indices: a rank-1 tensor of type int32 / int64
      fields: (optional) list of fields to also gather from.  If None (default),
        all fields are gathered from.  Pass an empty fields list to only gather
        the box coordinates.
      scope: name scope.

    Returns:
      subboxlist: a BoxList corresponding to the subset of the input BoxList
      specified by indices
    Raises:
      ValueError: if specified field is not contained in boxlist or if the
        indices are not of type int32
    """
    with tf.name_scope(scope, 'Gather'):
        if len(indices.shape.as_list()) != 1:
            raise ValueError('indices should have rank 1')
        if indices.dtype != tf.int32 and indices.dtype != tf.int64:
            raise ValueError('indices should be an int32 / int64 tensor')
        subboxlist = BoxList(tf.gather(boxlist.boxes, indices))
        if fields is None:
            fields = boxlist.get_extra_fields()
        for field in fields:
            if not boxlist.has_field(field):
                raise ValueError('boxlist must contain all specified fields')
            subfieldlist = tf.gather(boxlist.get_field(field), indices)
            subboxlist.add_field(field, subfieldlist)
        for tracking in boxlist.get_all_trackings():
            subboxlist.set_tracking(tracking, boxlist.get_tracking(tracking))
        return subboxlist


def concatenate(boxlists, fields=None, scope=None):
    """Concatenate list of BoxLists.

    This op concatenates a list of input BoxLists into a larger BoxList.  It also
    handles concatenation of BoxList fields as long as the field tensor shapes
    are equal except for the first dimension.

    Args:
      boxlists: list of BoxList objects
      fields: optional list of fields to also concatenate.  By default, all
        fields from the first BoxList in the list are included in the
        concatenation.
      scope: name scope.

    Returns:
      BoxList
    Raises:
      ValueError: if boxlists is invalid (i.e., is not a list, is empty, or
        contains non BoxList objects), or if requested fields are not contained in
        all boxlists
    """
    with tf.name_scope(scope, 'Concatenate'):
        if not isinstance(boxlists, list):
            raise ValueError('boxlists should be a list')
        if not boxlists:
            raise ValueError('boxlists should have nonzero length')
        for boxlist in boxlists:
            if not isinstance(boxlist, BoxList):
                raise ValueError('all elements of boxlists should be BoxList objects')
        concatenated = BoxList(
            tf.concat([boxlist.boxes for boxlist in boxlists], 0))
        if fields is None:
            fields = boxlists[0].get_extra_fields()
        for field in fields:
            first_field_shape = boxlists[0].get_field(field).get_shape().as_list()
            first_field_shape[0] = -1
            if None in first_field_shape:
                raise ValueError('field %s must have fully defined shape except for the'
                                 ' 0th dimension.' % field)
            for boxlist in boxlists:
                if not boxlist.has_field(field):
                    raise ValueError('boxlist must contain all requested fields')
                field_shape = boxlist.get_field(field).get_shape().as_list()
                field_shape[0] = -1
                if field_shape != first_field_shape:
                    raise ValueError('field %s must have same shape for all boxlists '
                                     'except for the 0th dimension.' % field)
            concatenated_field = tf.concat(
                [boxlist.get_field(field) for boxlist in boxlists], 0)
            concatenated.add_field(field, concatenated_field)
        return concatenated


def sort_by_field(boxlist, field, order=SortOrder.DESCENDING, scope=None):
    """Sort boxes and associated fields according to a scalar field.

    A common use case is reordering the boxes according to descending scores.

    Args:
        boxlist: BoxList holding N boxes.
        field: A BoxList field for sorting and reordering the BoxList.
        order: (Optional) descend or ascend. Default is descend.
        scope: name scope.

    Returns:
        sorted_boxlist: A sorted BoxList with the field in the specified order.

    Raises:
        ValueError: if specified field does not exist
        ValueError: if the order is not either descend or ascend
    """
    with tf.name_scope(scope, 'SortByField'):
        if order != SortOrder.DESCENDING and order != SortOrder.ASCENDING:
            raise ValueError('Invalid sort order')

        field_to_sort = boxlist.get_field(field)
        if len(field_to_sort.shape.as_list()) != 1:
            raise ValueError('Field should have rank 1')

        num_boxes = tf.shape(boxlist.boxes)[0]
        num_entries = tf.size(field_to_sort)
        length_assert = tf.Assert(
            tf.equal(num_boxes, num_entries),
            ['Incorrect field size: actual vs expected.', num_entries, num_boxes])

        with tf.control_dependencies([length_assert]):
            _, sorted_indices = tf.nn.top_k(field_to_sort, num_boxes, sorted=True)

        if order == SortOrder.ASCENDING:
            sorted_indices = tf.reverse_v2(sorted_indices, [0])

        return gather(boxlist, sorted_indices)


def filter_field_value_equals(boxlist, field, value, scope=None):
    """Filter to keep only boxes with field entries equal to the given value.

    Args:
      boxlist: BoxList holding N boxes.
      field: field name for filtering.
      value: scalar value.
      scope: name scope.

    Returns:
      a BoxList holding M boxes where M <= N

    Raises:
      ValueError: if boxlist not a BoxList object or if it does not have
        the specified field.
    """
    with tf.name_scope(scope, 'FilterFieldValueEquals'):
        if not isinstance(boxlist, BoxList):
            raise ValueError('boxlist must be a BoxList')
        if not boxlist.has_field(field):
            raise ValueError('boxlist must contain the specified field')
        filter_field = boxlist.get_field(field)
        gather_index = tf.reshape(tf.where(tf.equal(filter_field, value)), [-1])
        return gather(boxlist, gather_index)


def filter_scores_greater_than(boxlist, thresh, scope=None):
    """Filter to keep only boxes with score exceeding a given threshold.

    This op keeps the collection of boxes whose corresponding scores are
    greater than the input threshold.

    Args:
      boxlist: BoxList holding N boxes.  Must contain a 'scores' field
        representing detection scores.
      thresh: scalar threshold
      scope: name scope.

    Returns:
      a BoxList holding M boxes where M <= N

    Raises:
      ValueError: if boxlist not a BoxList object or if it does not
        have a scores field
    """
    with tf.name_scope(scope, 'FilterGreaterThan'):
        if not isinstance(boxlist, BoxList):
            raise ValueError('boxlist must be a BoxList')
        if not boxlist.has_field('scores'):
            raise ValueError('input boxlist must have \'scores\' field')
        scores = boxlist.get_field('scores')
        if len(scores.shape.as_list()) > 2:
            raise ValueError('Scores should have rank 1 or 2')
        if len(scores.shape.as_list()) == 2 and scores.shape.as_list()[1] != 1:
            raise ValueError('Scores should have rank 1 or have shape '
                             'consistent with [None, 1]')
        high_score_indices = tf.cast(tf.reshape(
            tf.where(tf.greater(scores, thresh)),
            [-1]), tf.int32)
        return gather(boxlist, high_score_indices)


def _copy_extra_datas(boxlist_to_copy_to, boxlist_to_copy_from):
    """Copies the extra fields of boxlist_to_copy_from to boxlist_to_copy_to.

    Args:
      boxlist_to_copy_to: BoxList to which extra fields are copied.
      boxlist_to_copy_from: BoxList from which fields are copied.

    Returns:
      boxlist_to_copy_to with extra fields.
    """
    for field in boxlist_to_copy_from.get_extra_fields():
        boxlist_to_copy_to.add_field(field, boxlist_to_copy_from.get_field(field))
    for tracking in boxlist_to_copy_from.get_all_trackings():
        boxlist_to_copy_to.set_tracking(tracking, boxlist_to_copy_from.get_tracking(tracking))
    return boxlist_to_copy_to


def to_normalized_coordinates(boxlist, image_shape, check_range=False, scope=None):
    """Converts absolute box coordinates to normalized coordinates in [0, 1].

    Usually one uses the dynamic shape of the image or conv-layer tensor:
      boxlist = boxlist_ops.to_normalized_coordinates(boxlist,
                                                      tf.shape(images)[1],
                                                      tf.shape(images)[2]),

    This function raises an assertion failed error at graph execution time when
    the maximum coordinate is smaller than 1.01 (which means that coordinates are
    already normalized). The value 1.01 is to deal with small rounding errors.

    Args:
      boxlist: BoxList with coordinates in terms of pixel-locations.
      height: Maximum value for height of absolute box coordinates.
      width: Maximum value for width of absolute box coordinates.
      check_range: If True, checks if the coordinates are normalized or not.
      scope: name scope.

    Returns:
      boxlist with normalized coordinates in [0, 1].
    """
    with tf.name_scope(scope, 'ToNormalizedCoordinates'):
        height = tf.cast(image_shape[0], tf.float32)
        width = tf.cast(image_shape[1], tf.float32)

        if check_range:
            max_val = tf.reduce_max(boxlist.boxes)
            max_assert = tf.Assert(tf.greater(max_val, 1.01),
                                   ['max value is lower than 1.01: ', max_val])
            with tf.control_dependencies([max_assert]):
                width = tf.identity(width)

        return scale(boxlist, 1 / height, 1 / width)


def to_absolute_coordinates(boxlist,
                            image_shape,
                            check_range=True,
                            maximum_normalized_coordinate=1.01,
                            scope=None):
    """Converts normalized box coordinates to absolute pixel coordinates.

    This function raises an assertion failed error when the maximum box coordinate
    value is larger than maximum_normalized_coordinate (in which case coordinates
    are already absolute).

    Args:
      boxlist: BoxList with coordinates in range [0, 1].
      height: Maximum value for height of absolute box coordinates.
      width: Maximum value for width of absolute box coordinates.
      check_range: If True, checks if the coordinates are normalized or not.
      maximum_normalized_coordinate: Maximum coordinate value to be considered
        as normalized, default to 1.01.
      scope: name scope.

    Returns:
      boxlist with absolute coordinates in terms of the image size.

    """
    with tf.name_scope(scope, 'ToAbsoluteCoordinates'):
        height = tf.cast(image_shape[0], tf.float32)
        width = tf.cast(image_shape[1], tf.float32)

        # Ensure range of input boxes is correct.
        if check_range:
            box_maximum = tf.reduce_max(boxlist.boxes)
            max_assert = tf.Assert(
                tf.greater_equal(maximum_normalized_coordinate, box_maximum),
                ['maximum box coordinate value is larger '
                 'than %f: ' % maximum_normalized_coordinate, box_maximum])
            with tf.control_dependencies([max_assert]):
                width = tf.identity(width)
        return scale(boxlist, height, width)


def pad_or_clip_boxlist(boxlist, num_boxes, scope=None):
    """Pads or clips all fields of a BoxList.

    Args:
        boxlist: A BoxList with arbitrary of number of boxes.
        num_boxes: First num_boxes in boxlist are kept.
        The fields are zero-padded if num_boxes is bigger than the
        actual number of boxes.
        scope: name scope.

    Returns:
        BoxList with all fields padded or clipped.
    """
    with tf.name_scope(scope, 'PadOrClipBoxList'):
        subboxlist = BoxList(shape_utils.pad_or_clip_tensor(
            boxlist.boxes, num_boxes))
        for field in boxlist.get_extra_fields():
            subfield = shape_utils.pad_or_clip_tensor(
                boxlist.get_field(field), num_boxes)
            subboxlist.add_field(field, subfield)
        for tracking in boxlist.get_all_trackings():
            subboxlist.set_tracking(tracking, boxlist.get_tracking(tracking))
        return subboxlist
