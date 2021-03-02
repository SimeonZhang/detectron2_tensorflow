import tensorflow as tf

from ..layers import resize_images
from ..structures import box_list
from ..structures import box_list_ops
from ..structures import mask_ops


def detector_postprocess(
    results, output_shape, mask_format, image_shapes=None, mask_threshold=0.5, scope=None
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.
    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.
    Args:
        results (SparseBoxList): the raw outputs from the detector.
            `results.get_tracking('image_shape')` contains the input image resolution
            the detector sees.
            This object might be modified in-place.
        output_shape: the desired output resolution.
        mask_format: raw, fixed or conventional
    Returns:
        Boxlist: the resized output from the model, based on the output resolution
    """
    if not results.has_field("pred_masks"):
        return results
    with tf.name_scope(scope, "DetectorPostprocess"):

        if mask_format in ["conventional", "fixed"]:
            results = box_list.SparseBoxList.from_dense(results)
            box_masks = results.data.get_field("pred_masks")
            if mask_format == "fixed":
                assert image_shapes is not None, (
                    "Detection results should carry the true input shape.")
                image_shapes = results.get_tracking('image_shape')
                scales = tf.expand_dims(output_shape, axis=0) / image_shapes
                scales = tf.gather(scales, results.indices[:, 0])
                boxes = box_list_ops.scale(
                    results.data, scales[:, 0:1], scales[:, 1:2]
                ).boxes
            else:
                boxes = results.data.boxes
            pred_masks = mask_ops.reframe_box_masks_to_image_masks(
                box_masks, boxes, output_shape, mask_threshold
            )
            results.data.set_field("pred_masks", pred_masks)
            results = results.to_dense()
        elif mask_format == "raw":
            pred_masks = results.get_field("pred_masks")
            pred_masks = tf.cast(tf.greater(image_masks, mask_threshold), tf.uint8)
            results.set_field("pred_masks", pred_masks)
        else:
            raise ValueError(f"mask format '{mask_format}' is not recognized.")

        return results


def sem_seg_postprocess(result, image_shapes, output_shape, mask_format, scope=None):
    """
    Return semantic segmentation predictions in the original resolution.
    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.
    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (N, H, W, C),
            where C is the number of classes, and H, W are the height and width of the prediction.
        image_shapes (Tensor): [N, 2], image size that segmentor is taking as input.
        output_shape: the desired output resolution.
    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (N, output_shape[0], output_shape[1], C) that contains per-pixel soft predictions.
    """
    with tf.name_scope(scope, "SemSegPostprocess"):

        def postprocess_per_image(args):
            sem_seg, true_shape = args
            if mask_format == "fixed":
                sem_seg = resize_images(
                    sem_seg[:true_shape[0], :true_shape[1], ...], output_shape
                )
            else:
                sem_seg = tf.pad(
                    sem_seg[:true_shape[0], :true_shape[1], ...],
                    [[0, output_shape[0]-true_shape[0]], [0, output_shape[1]-true_shape[1]], [0, 0]]
                )

            return sem_seg

        result = tf.map_fn(postprocess_per_image, [result, image_shapes], dtype=result.dtype)
        return result
