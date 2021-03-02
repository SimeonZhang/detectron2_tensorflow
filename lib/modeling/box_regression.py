import math
import tensorflow as tf

from ..utils import shape_utils

# Value for clamping large dw and dh predictions. The heuristic is that we
# clamp such that dw and dh are no larger than what would transform
# a 16px box into a 1000px box
# based on a small anchor, 16px, and a typical image size, 1000px.
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


__all__ = ["Box2BoxTransform", "Box2BoxTransformRotated"]


class Box2BoxTransform(object):
    """
    The box-to-box transform defined in R-CNN. The transformation is
    parameterized by 4 deltas: (dy, dx, dh, dw). The transformation scales
    the box's height and width by exp(dh), exp(dw) and shifts a box's center
    by the offset (dy * height, dx * width).
    """

    def __init__(self, weights, scale_clamp=_DEFAULT_SCALE_CLAMP):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally
                set such that the deltas have unit variance; now they are
                treated as hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box
                scaling factors (dw and dh) are clamped such that they are
                <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be
        used to transform the `src_boxes` into the `target_boxes`. That is,
        the relation ``target_boxes == self.apply_deltas(deltas, src_boxes)``
        is true (unless any delta is too large and is clamped).
        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g.,
                ground-truth boxes.
        """
        with tf.name_scope("BoxEncoding"):
            assert isinstance(src_boxes, tf.Tensor), type(src_boxes)
            assert isinstance(target_boxes, tf.Tensor), type(target_boxes)

            src_heights = src_boxes[:, 2] - src_boxes[:, 0]
            src_widths = src_boxes[:, 3] - src_boxes[:, 1]
            src_ctr_y = src_boxes[:, 0] + 0.5 * src_heights
            src_ctr_x = src_boxes[:, 1] + 0.5 * src_widths

            target_heights = target_boxes[:, 2] - target_boxes[:, 0]
            target_widths = target_boxes[:, 3] - target_boxes[:, 1]
            target_ctr_y = target_boxes[:, 0] + 0.5 * target_heights
            target_ctr_x = target_boxes[:, 1] + 0.5 * target_widths

            wy, wx, wh, ww = self.weights
            dy = wy * (target_ctr_y - src_ctr_y) / src_heights
            dx = wx * (target_ctr_x - src_ctr_x) / src_widths
            dh = wh * tf.log(target_heights / src_heights)
            dw = ww * tf.log(target_widths / src_widths)

            # assert_positive = tf.assert_positive(
            #     src_heights, src_heights,
            #     message="Input boxes to Box2BoxTransform are not valid!")
            # with tf.control_dependencies([assert_positive]):
            deltas = tf.stack([dy, dx, dh, dw], axis=1)
            return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dy, dx, dh, dw) to `boxes`.
        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where
                k >= 1. deltas[i] represents k potentially different
                class-specific box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        with tf.name_scope("BoxDecoding"):
            # try:
            #     is_finite = tf.is_finite(deltas)
            # except:
            #     is_finite = tf.math.is_finite(deltas)
            # assert_finite = tf.Assert(tf.reduce_all(is_finite), deltas)
            # with tf.control_dependencies([assert_finite]):
            #     boxes = tf.cast(boxes, deltas.dtype)
            deltas_shape = shape_utils.combined_static_and_dynamic_shape(deltas)

            heights = boxes[:, 2] - boxes[:, 0]
            widths = boxes[:, 3] - boxes[:, 1]
            ctr_y = boxes[:, 0] + 0.5 * heights
            ctr_x = boxes[:, 1] + 0.5 * widths
            # [N]

            wy, wx, wh, ww = self.weights
            deltas = tf.reshape(deltas, [deltas_shape[0], deltas_shape[1] // 4, 4])
            dy = deltas[:, :, 0] / wy
            dx = deltas[:, :, 1] / wx
            dh = deltas[:, :, 2] / wh
            dw = deltas[:, :, 3] / ww
            # [N, k]

            # Prevent sending too large values into tf.exp()
            dh = tf.minimum(dh, self.scale_clamp)
            dw = tf.minimum(dw, self.scale_clamp)

            pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
            pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
            pred_h = tf.exp(dh) * heights[:, None]
            pred_w = tf.exp(dw) * widths[:, None]

            y1 = pred_ctr_y - 0.5 * pred_h  # y1
            x1 = pred_ctr_x - 0.5 * pred_w  # x1
            y2 = pred_ctr_y + 0.5 * pred_h  # y2
            x2 = pred_ctr_x + 0.5 * pred_w  # x2
            pred_boxes = tf.stack([y1, x1, y2, x2], axis=2)
            return tf.reshape(pred_boxes, deltas_shape)


# class Box2BoxTransformRotated(object):
#     """
#     The box-to-box transform defined in Rotated R-CNN. The transformation is
#     parameterized by 5 deltas: (dx, dy, dw, dh, da). The transformation
#     scales the box's width and height by exp(dw), exp(dh), shifts a box's
#     center by the offset (dx * width, dy * height), and rotate a box's angle
#     by da (radians).
#     Note: angles of deltas are in radians while angles of boxes are in degrees.
#     """

#     def __init__(self, weights, scale_clamp=_DEFAULT_SCALE_CLAMP):
#         """
#         Args:
#             weights (5-element tuple): Scaling factors that are applied to the
#                 (dx, dy, dw, dh, da) deltas. These are treated as
#                 hyperparameters of the system.
#             scale_clamp (float): When predicting deltas, the predicted box
#                 scaling factors (dw and dh) are clamped such that they are
#                 <= scale_clamp.
#         """
#         self.weights = weights
#         self.scale_clamp = scale_clamp

#     @staticmethod
#     def rotate_angles(angles):
#         def rotate_clockwise(a):
#             mask = a < -180.0
#             a += 360.0 * tf.cast(mask, a.dtype)
#             return a
#         angles = tf.while_loop(
#             lambda a: tf.reduce_any(a < -180.0), rotate_clockwise, [angles])

#         def rotate_anticlockwise(a):
#             mask = a > 180.0
#             a -= 360.0 * tf.cast(mask, a.dtype)
#             return a
#         angles = tf.while_loop(
#             lambda a: tf.reduce_any(a > 180.0), rotate_anticlockwise, [angles])
#         return angles

#     def get_deltas(self, src_boxes, target_boxes):
#         """
#         Get box regression transformation deltas (dx, dy, dw, dh, da) that can
#         be used to transform the `src_boxes` into the `target_boxes`. That is,
#         the relation ``target_boxes == self.apply_deltas(deltas, src_boxes)``
#         is true (unless any delta is too large and is clamped).
#         Args:
#             src_boxes (Tensor): Nx5 source boxes, e.g., object proposals
#             target_boxes (Tensor): Nx5 target of the transformation, e.g.,
#                 ground-truth boxes.
#         """
#         assert isinstance(src_boxes, tf.Tensor), type(src_boxes)
#         assert isinstance(target_boxes, tf.Tensor), type(target_boxes)

#         src_ctr_x, src_ctr_y, src_widths, src_heights, src_angles = (
#             tf.unstack(src_boxes, axis=1))

#         (target_ctr_x, target_ctr_y, target_widths, target_heights,
#             target_angles) = tf.unstack(target_boxes, axis=1)

#         wx, wy, ww, wh, wa = self.weights
#         dx = wx * (target_ctr_x - src_ctr_x) / src_widths
#         dy = wy * (target_ctr_y - src_ctr_y) / src_heights
#         dw = ww * tf.log(target_widths / src_widths)
#         dh = wh * tf.log(target_heights / src_heights)
#         # Angles of deltas are in radians while angles of boxes are in degrees.
#         # the conversion to radians serve as a way to normalize the values
#         da = target_angles - src_angles

#         da = self.rotate_angles(da)
#         da *= wa * math.pi / 180.0

#         assert_positive = tf.assert_positive(
#             src_widths, src_widths,
#             message="Input boxes to Box2BoxTransformRotated are not valid!")
#         with tf.control_dependencies([assert_positive]):
#             deltas = tf.stack((dx, dy, dw, dh), axis=1)
#         return deltas

#     def apply_deltas(self, deltas, boxes):
#         """
#         Apply transformation `deltas` (dx, dy, dw, dh, da) to `boxes`.
#         Args:
#             deltas (Tensor): transformation deltas of shape (N, 5).
#                 deltas[i] represents box transformation for the single box
#                 boxes[i].
#             boxes (Tensor): boxes to transform, of shape (N, 5)
#         """
#         assert tf_utils.shape(deltas)[1] == 5 and tf_utils.shape(boxes)[1] == 5
#         try:
#             is_finite = tf.is_finite(deltas)
#         except:
#             is_finite = tf.math.is_finite(deltas)
#         assert_finite = tf.Assert(tf.reduce_all(is_finite), deltas)
#         with tf.control_dependencies([assert_finite]):
#             boxes = tf.cast(boxes, deltas.dtype)
#         deltas_shape = tf_utils.shape(deltas)

#         ctr_x, ctr_y, widths, heights, angles = tf.unstack(boxes, axis=1)
#         wx, wy, ww, wh, wa = self.weights
#         dx, dy, dw, dh, da = tf.unstack(deltas, axis=1)

#         dx /= wx
#         dy /= wy
#         dw /= ww
#         dh /= wh
#         da /= wa

#         # Prevent sending too large values into torch.exp()
#         dw = tf.minimum(dw, self.scale_clamp)
#         dh = tf.minimum(dh, self.scale_clamp)

#         pred_ctr_x = dx * widths + ctr_x  # x_ctr
#         pred_ctr_y = dy * heights + ctr_y  # y_ctr
#         pred_w = tf.exp(dw) * widths  # width
#         pred_h = tf.exp(dh) * heights  # height

#         # Following original RRPN implementation,
#         # angles of deltas are in radians while angles of boxes are in degrees.
#         pred_angle = da * 180.0 / math.pi + angles
#         pred_angle = self.rotate_angles(pred_angle)

#         pred_boxes = tf.stack(
#             [pred_ctr_x, pred_ctr_y, pred_w, pred_h, pred_angle], axis=1)

#         return pred_boxes
