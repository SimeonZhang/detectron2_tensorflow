import tensorflow as tf

from ..layers import resize_images
from ..data import fields
from ..structures import box_list
from ..structures import box_list_ops
from ..utils import id_utils


def detector_postprocess(
    results,
    input_shape,
    thing_class_names,
    label_offset=1
):
    """
    Normalize the boxes related to the image size.
    Assign class names for each box.
    Args:
        results: the raw output dict from the detector.
            This object might be updated in-place.
        input_shape: the image shape the input image resolution the detector sees.
        thing_class_names: thing class names.
    Returns:
        results: the resized output from the model, based on the output resolution.
    """
    result_fields = fields.ResultFields
    num_instances = tf.reduce_sum(
        tf.cast(results[result_fields.is_valid], tf.int32), axis=1)
    max_num_instances = tf.reduce_max(num_instances)

    def detector_postprocess_single_image(args):
        results, input_shape = args

        boxes = results[result_fields.boxes]
        classes = results[result_fields.classes]
        scores = results[result_fields.scores]
        class_names = tf.gather(thing_class_names, classes)

        result_boxlist = box_list.BoxList(boxes)
        result_boxlist.add_field(result_fields.class_names, class_names)
        result_boxlist.add_field(result_fields.classes, classes + label_offset)
        result_boxlist.add_field(result_fields.scores, scores)

        if result_fields.masks in results:
            result_boxlist.add_field(result_fields.masks, results[result_fields.masks])
        
        result_boxlist = box_list_ops.boolean_mask(
            result_boxlist, results[result_fields.is_valid]
        )

        result_boxlist = box_list_ops.to_normalized_coordinates(
            result_boxlist, input_shape, check_range=False)
        result_boxlist = box_list_ops.sort_by_field(result_boxlist, result_fields.scores)
        result_boxlist = box_list_ops.pad_or_clip_boxlist(result_boxlist, max_num_instances)
        results = result_boxlist.as_tensor_dict()
        return results

    expected_fields = [
        result_fields.boxes,
        result_fields.classes,
        result_fields.scores,
        result_fields.masks
    ]
    dtype = {k: v.dtype for k, v in results.items() if k in expected_fields}
    dtype[result_fields.class_names] = tf.string
    results = tf.map_fn(
        detector_postprocess_single_image, [results, input_shape], dtype=dtype
    )
    results["num_detections"] = num_instances
    return results


def sem_seg_postprocess(
    sem_seg,
    input_shape,
    output_shape,
    stuff_class_names,
    stuff_ignore_value=None,
    stuff_area_limit=0.001
):
    """
    Return semantic segmentation predictions in the original resolution.
    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.
    Args:
        sem_seg (Tensor): semantic segmentation prediction logits. A tensor of shape (N, H, W),
            where H, W are the height and width of the prediction.
        input_shape (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.
    Returns:
        results: the resized output from the model, based on the output resolution.
    """
    serving_fields = fields.ServingFields
    num_classes = len(stuff_class_names)
    stuff_included = [True] * num_classes
    stuff_included[0] = False
    stuff_included[-1] = False
    if stuff_ignore_value is not None and stuff_ignore_value < num_classes:
        stuff_included[stuff_ignore_value] = False

    def sem_seg_postprocess_single_image(args):
        sem_seg, input_shape, output_shape = args
        sem_seg = sem_seg[:input_shape[0], :input_shape[1]]
        one_hot_sem_seg = tf.one_hot(sem_seg, num_classes)

        total_area = tf.cast(input_shape[0] * input_shape[1], tf.float32)
        area_per_class = tf.cast(tf.reduce_sum(one_hot_sem_seg, [0, 1]), tf.float32)
        area_per_class = tf.divide(area_per_class, total_area)

        show_up_mask = tf.logical_and(
            tf.greater(area_per_class, stuff_area_limit), stuff_included
        )
        classes = tf.boolean_mask(tf.range(num_classes), show_up_mask)
        class_names = tf.boolean_mask(stuff_class_names, show_up_mask)
        areas = tf.boolean_mask(area_per_class, show_up_mask)

        padding_end = num_classes - tf.shape(classes)[0]
        one_hot_sem_seg = resize_images(
            one_hot_sem_seg, output_shape, align_corners=True
        )
        one_hot_sem_seg = one_hot_sem_seg * tf.cast(show_up_mask, tf.float32)
        one_hot_sem_seg = tf.round(one_hot_sem_seg)
        sem_seg = tf.reduce_sum(
            one_hot_sem_seg * tf.range(num_classes, dtype=tf.float32), axis=-1
        )
        sem_seg = id_utils.id2rgb(sem_seg)
        sem_seg = tf.image.encode_png(tf.cast(sem_seg, tf.uint8))
        classes = tf.pad(classes, [[0, padding_end]])
        class_names = tf.pad(class_names, [[0, padding_end]])
        areas = tf.pad(areas, [[0, padding_end]])
        results = {}
        results[serving_fields.sem_seg] = sem_seg
        results[serving_fields.sem_seg_classes] = classes
        results[serving_fields.sem_seg_class_names] = class_names
        results[serving_fields.sem_seg_areas] = areas
        return results

    dtype = {
        serving_fields.sem_seg: tf.string,
        serving_fields.sem_seg_classes: tf.int64,
        serving_fields.sem_seg_class_names: tf.string,
        serving_fields.sem_seg_areas: tf.float32
    }

    results = tf.map_fn(
        sem_seg_postprocess_single_image, [sem_seg, input_shape, output_shape], dtype=dtype
    )
    return results


def panoptic_postprocess(
    segments_info,
    input_shape,
    thing_contiguous_id_to_dataset_id,
    stuff_contiguous_id_to_dataset_id,
    class_names,
):
    """
    Postprocess the panoptic results and return results in detection manner.
    return:
        results: see `detector_postprocess`.
    """
    result_fields = fields.ResultFields

    def get_detection_like_result_single_image(segments_info):
        
        instance_dict = {
            result_fields.boxes: segments_info["bbox"],
            result_fields.classes: segments_info["category_id"],
            result_fields.scores: segments_info["score"],
            result_fields.is_valid: segments_info["is_valid"]
        }
        boxlist = box_list.BoxList.from_tensor_dict(instance_dict)
        
        isthing = segments_info["isthing"]
        thing_boxlist = box_list_ops.boolean_mask(boxlist, isthing)
        stuff_boxlist = box_list_ops.boolean_mask(boxlist, tf.logical_not(isthing))

        thing_classes = tf.gather(thing_contiguous_id_to_dataset_id, thing_boxlist.get_field(result_fields.classes))
        stuff_classes = tf.gather(stuff_contiguous_id_to_dataset_id, stuff_boxlist.get_field(result_fields.classes))
        thing_classes = tf.cast(thing_classes, tf.int64)
        stuff_classes = tf.cast(stuff_classes, tf.int64)
        thing_boxlist.set_field(result_fields.classes, thing_classes)
        stuff_boxlist.set_field(result_fields.classes, stuff_classes)

        boxlist = box_list_ops.concatenate([thing_boxlist, stuff_boxlist])
        return boxlist.as_tensor_dict()

    dtype = {
        result_fields.boxes: tf.float32,
        result_fields.classes: tf.int64,
        result_fields.scores: tf.float32,
        result_fields.is_valid: tf.bool
    }
    detection_like_result = tf.map_fn(
        get_detection_like_result_single_image, segments_info, dtype=dtype
    )
    return detector_postprocess(detection_like_result, input_shape, class_names)
