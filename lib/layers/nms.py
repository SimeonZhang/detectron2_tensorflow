import tensorflow as tf

from ..utils.shape_utils import combined_static_and_dynamic_shape


def batch_nms(
    boxes,
    scores,
    max_output_size,
    axis=0,
    iou_threshold=0.5,
    scope=None
):
    with tf.name_scope(scope, "BatchNMS", [boxes, scores]):
        assert len(boxes.get_shape()) == 3
        assert len(scores.get_shape()) == 2
        assert axis in [0, 1]
        if axis == 0:
            boxes = tf.transpose(boxes, [1, 0, 2])
            scores = tf.transpose(scores, [1, 0])

        def fn(X):
            keep = tf.image.non_max_suppression(
                X[0], X[1], max_output_size, iou_threshold)
            return keep
        return tf.map_fn(fn, [boxes, scores])


def matrix_nms(
    masks,
    classes,
    scores,
    sum_masks=None,
    kernel="gaussian",
    sigma=2.0,
    scope=None
):
    with tf.name_scope(scope, "MatrixNMS", [masks, classes, scores]):
        assert len(masks.get_shape()) == 3
        assert len(classes.get_shape()) == 1
        assert len(scores.get_shape()) == 1
        
        mask_shape = combined_static_and_dynamic_shape(masks)
        num_samples = mask_shape[0]
        if sum_masks is None:
            sum_masks = tf.reduce_sum(masks, [1, 2])
        masks = tf.reshape(masks, [num_samples, mask_shape[1]*mask_shape[2]])

        # iou
        inter_matrix = tf.matmul(masks, masks, transpose_b=True)
        sum_matrix = tf.tile(tf.expand_dims(sum_masks, axis=0), [num_samples, 1])
        union_matrix = sum_matrix + tf.transpose(sum_matrix) - inter_matrix

        iou_matrix = inter_matrix / union_matrix
        iou_matrix = iou_matrix - tf.matrix_band_part(iou_matrix, -1, 0)

        # class specific matrix
        class_matrix = tf.tile(tf.expand_dims(classes, axis=0), [num_samples, 1])
        class_spec_matrix = tf.equal(class_matrix, tf.transpose(class_matrix))
        class_spec_matrix = tf.cast(class_spec_matrix, tf.float32)
        # class_spec_matrix = class_spec_matrix - tf.matrix_band_part(class_spec_matrix, -1, 0)

        # IoU decay
        iou = iou_matrix * class_spec_matrix

        # IoU compensation
        compensate_iou = tf.reduce_max(iou, axis=0)
        compensate_iou = tf.tile(tf.expand_dims(compensate_iou, axis=0), [num_samples, 1])
        compensate_iou = tf.transpose(compensate_iou)

        # matrix nms
        if kernel == "gaussian":
            decay_matrix = tf.exp(-1 * sigma * (iou ** 2 - compensate_iou ** 2))
        elif kernel == "linear":
            decay_matrix = (1. - iou) / (1. - compensate_iou)
        else:
            raise NotImplementedError(f"NMS kernel {kernel} not implemented yet.")

        decay_factor = tf.reduce_min(decay_matrix, axis=0)

        # update the score
        updated_scores = scores * decay_factor
        return updated_scores
