"""
Contains classes specifying naming conventions used for the model.
Specifies:
  InputFields: standard fields used by reader/preprocessor/batcher.
  TfExampleFields: standard fields for tf-example data format.
"""


class InputFields(object):
    """Names for the input tensors.
    Holds the standard data field names to use for identifying input tensors. This
    should be used by the decoder to identify keys for the returned tensor_dict
    containing input tensors. And it should be used by the model to identify the
    tensors it needs.
    Attributes:
      image: image.
      filename: original filename of the dataset (without common path).
      groundtruth_classes: [max_gt_instances] groundtruth class int ids.
          might be zero padded.
      groundtruth_boxes: [max_gt_instances, 4], (ymin, xmin, ymax, xmax),
          groundtruth boxes in normalized coordinates. might be zero padded
      groundtruth_instance_masks: [max_gt_instances, image_height, image_width],
          groundtruth instance masks. might be zero padded.
      groundtruth_group_of: [max_gt_instances], int32, group of objects = 1,
          or not = 0.
      groundtruth_difficult: [max_gt_instances], int32, difficult objects = 1,
          or not = 0.
      num_groundtruth_boxes: int32, number of groundtruth boxes.
    """
    image = 'image'
    sem_seg = 'sem_seg'
    filename = 'filename'
    key = 'key'
    orig_shape = 'orig_shape'
    true_shape = 'true_shape'
    gt_masks = 'gt_masks'
    gt_boxes = 'gt_boxes'
    gt_classes = 'gt_classes'
    gt_is_crowd = 'gt_is_crowd'
    gt_difficult = 'gt_difficult'
    is_valid = 'is_valid'


class TfExampleFields(object):
    """TF-example proto feature names for object detection.
    Holds the standard feature names to load from an Example proto for object
    detection.
    Attributes:
      image_encoded: JPEG encoded string
      image_format: image format, e.g. "JPEG"
      filename: filename
      key: hash key
      height: height of image in pixels, e.g. 462
      width: width of image in pixels, e.g. 581
      channel: number of color channels.
      object_class_text: labels in text format, e.g. ["person", "cat"]
      object_class_label: labels in numbers, e.g. [16, 8]
      object_bbox_xmin: xmin coordinates of groundtruth box, e.g. 10, 30
      object_bbox_xmax: xmax coordinates of groundtruth box, e.g. 50, 40
      object_bbox_ymin: ymin coordinates of groundtruth box, e.g. 40, 50
      object_bbox_ymax: ymax coordinates of groundtruth box, e.g. 80, 70
      object_difficult: is this object's class annotation reliable.
      object_is_crowd: is this a group of objects or not.
    """
    image_encoded = 'image/encoded'
    sem_seg = 'image/sem_seg'
    image_format = 'image/format'
    filename = 'filename'
    key = 'image/key'
    height = 'image/height'
    width = 'image/width'
    object_class_text = 'image/object/class/text'
    object_class_label = 'image/object/class/label'
    instance_masks = 'image/object/mask'
    object_bbox_ymin = 'image/object/bbox/ymin'
    object_bbox_xmin = 'image/object/bbox/xmin'
    object_bbox_ymax = 'image/object/bbox/ymax'
    object_bbox_xmax = 'image/object/bbox/xmax'
    object_difficult = 'image/object/difficult'
    object_is_crowd = 'image/object/is_crowd'


class ResultFields(object):
    boxes = "boxes"
    classes = "classes"
    class_names = "class_names"
    scores = "scores"
    masks = "masks"
    is_valid = "is_valid"
    sem_seg = "sem_seg"
    panoptic_seg = "panoptic_seg"


class ServingFields(object):
    boxes = "boxes"
    classes = "classes"
    scores = "scores"
    sem_seg_class_names = "class_names"
