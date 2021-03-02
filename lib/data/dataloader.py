import os
import functools
import tensorflow as tf

from . import transform
from . import fields
from ..utils import tf_utils
from ..utils import id_utils


def build_dataloader(cfg, training=None):
    """build a dataset composing of tensor_dict.
    Args:
        cfg:
        training:

    Return:
        dataloader: a tf.data.Iterator composing of tensor_dict.
    """
    if training is None:
        training = tf_utils.get_training_phase()
    root_dir = cfg.DATASETS.ROOT_DIR
    split_name = cfg.DATASETS.TRAIN if training else cfg.DATASETS.VAL

    load_instance_masks = cfg.MODEL.MASK_ON
    load_semantic_mask = cfg.DATALOADER.LOAD_SEMANTIC_MASKS

    sample_1_of_n = 1 if training else cfg.DATALOADER.SAMPLE_1_OF_N

    batch_size = cfg.SOLVER.IMS_PER_GPU
    num_parallel_calls = batch_size * cfg.DATALOADER.NUM_PARALLEL_BATCHES
    num_prefetch_batches = cfg.DATALOADER.NUM_PREFETCH_BATCHES

    file_pattern = os.path.join(root_dir, "{}*.tfrecord".format(split_name))

    def process_fn(value):
        tensor_dict = parse_tf_example(
            value,
            load_instance_masks,
            load_semantic_mask
        )

        tensor_dict = transform.run(
            cfg,
            tensor_dict,
            training,
            load_instance_masks,
            load_semantic_mask
        )
        return tensor_dict
    with tf.name_scope("DataLoader"):
        dataset = read_dataset(cfg, file_pattern, training)
        if sample_1_of_n > 1:
            dataset = dataset.shard(sample_1_of_n, 0)

        dataset = dataset.map(process_fn, num_parallel_calls=num_parallel_calls)

        batch_func = get_batch_fn(
            batch_size,
            load_instance_masks,
            load_semantic_mask
        )
        dataset = dataset.apply(batch_func)
        dataset = dataset.prefetch(num_prefetch_batches)

    return dataset.make_one_shot_iterator()


def read_dataset(cfg, file_pattern, training=False):
    """Reads a dataset, and handles repetition and shuffling.
    Args:
        file_read_func: Function to use in tf.contrib.data.parallel_interleave, to
        read every individual file into a tf.data.Dataset.
        file_pattern: A file pattern to match.
        config: A input_reader_builder.InputReader object.
    Returns:
        A tf.data.Dataset of (undecoded) tf-records based on config.
    """
    # Shard, shuffle, and read files.
    num_readers = cfg.DATALOADER.NUM_READERS if training else 1
    shuffle = cfg.DATALOADER.SHUFFLE if training else False
    num_epochs = None  # if training else 1

    filenames = tf.gfile.Glob(file_pattern)
    if num_readers > len(filenames):
        num_readers = len(filenames)
        tf.logging.warning(
            'num_readers has been reduced to %d to match input file shards.' % num_readers)
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
        filename_dataset = filename_dataset.shuffle(
            cfg.DATALOADER.FILENAME_SHUFFLE_BUFFER_SIZE
        )
    elif num_readers > 1:
        tf.logging.warning(
            '`shuffle` is false, but the input data stream is '
            'still slightly shuffled since `num_readers` > 1.')
    filename_dataset = filename_dataset.repeat(num_epochs)
    file_read_func = functools.partial(
        tf.data.TFRecordDataset, buffer_size=cfg.DATALOADER.FILE_READ_BUFFER_SIZE
    )
    records_dataset = filename_dataset.apply(
        tf.data.experimental.parallel_interleave(
            file_read_func,
            cycle_length=num_readers,
            block_length=cfg.DATALOADER.READ_BLOCK_LENGTH,
            sloppy=shuffle,
        )
    )
    if shuffle:
        records_dataset = records_dataset.shuffle(cfg.DATALOADER.SHUFFLE_BUFFER_SIZE)
    return records_dataset


def get_batch_fn(
    batch_size,
    load_instance_masks=False,
    load_semantic_mask=False
):
    def key_func(tensor_dict):
        image_shape = tensor_dict['true_shape']
        return tf.cast(image_shape[0] > image_shape[1], tf.int64)
    input_fields = fields.InputFields
    padding_shapes = {
        input_fields.image: [-1, -1, 3],
        input_fields.filename: [],
        input_fields.key: [],
        input_fields.orig_shape: [2],
        input_fields.true_shape: [2],
        input_fields.gt_boxes: [-1, 4],
        input_fields.gt_classes: [-1],
        input_fields.gt_difficult: [-1],
        input_fields.gt_is_crowd: [-1],
        input_fields.is_valid: [-1],
    }
    if load_instance_masks:
        padding_shapes[input_fields.gt_masks] = [-1, -1, -1]
    if load_semantic_mask:
        padding_shapes[input_fields.sem_seg] = [-1, -1]

    def reduce_func(key, elements):
        return elements.padded_batch(batch_size, padding_shapes)
    return tf.data.experimental.group_by_window(
        key_func, reduce_func, window_size=batch_size
    )


def parse_tf_example(
    serialized_example,
    load_instance_masks=False,
    load_semantic_mask=False
):
    """Decodes serialized tensorflow example and returns a tensor dictionary.

    Args:
      serialized_example: scalar Tensor tf.string containing a serialized Example
        protocol buffer.

    Returns:
      A dictionary of the following tensors.
      fields.InputFields.image - 3D uint8 tensor of shape [None, None, 3]
        containing image.
      fields.InputFields.original_shape - 1D int32 tensor of shape [2]
        containing shape of the image.
      fields.InputFields.filename - string tensor with original dataset
        filename.
      fields.InputFields.key - string tensor with unique sha256 hash key.
      fields.InputFields.groundtruth_boxes - 2D float32 tensor of shape
        [None, 4] containing box corners.
      fields.InputFields.groundtruth_classes - 1D int64 tensor of shape
        [None] containing classes for the boxes.
      fields.InputFields.groundtruth_difficult - 1D bool tensor of shape
        [None] indicating if the boxes represent `difficult` instances.
      fields.InputFields.groundtruth_group_of - 1D bool tensor of shape
        [None] indicating if the boxes represent `group_of` instances.
      fields.InputFields.groundtruth_instance_masks - 3D float32 tensor of
        shape [None, None, None] containing instance masks.
    """
    tfexample_fields = fields.TfExampleFields
    input_fields = fields.InputFields
    feature_map = {
        tfexample_fields.image_encoded:
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        tfexample_fields.image_format:
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        tfexample_fields.filename:
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        tfexample_fields.key:
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        tfexample_fields.height:
            tf.FixedLenFeature([], tf.int64, default_value=1),
        tfexample_fields.width:
            tf.FixedLenFeature([], tf.int64, default_value=1),
        tfexample_fields.object_class_label:
            tf.VarLenFeature(dtype=tf.int64),
        tfexample_fields.object_class_text:
            tf.VarLenFeature(dtype=tf.string),
        tfexample_fields.object_bbox_ymin:
            tf.VarLenFeature(dtype=tf.float32),
        tfexample_fields.object_bbox_xmin:
            tf.VarLenFeature(dtype=tf.float32),
        tfexample_fields.object_bbox_ymax:
            tf.VarLenFeature(dtype=tf.float32),
        tfexample_fields.object_bbox_xmax:
            tf.VarLenFeature(dtype=tf.float32),
        tfexample_fields.object_is_crowd:
            tf.VarLenFeature(tf.int64),
        tfexample_fields.object_difficult:
            tf.VarLenFeature(tf.int64),
    }

    if load_instance_masks:
        feature_map[tfexample_fields.instance_masks] = tf.VarLenFeature(dtype=tf.string)

    if load_semantic_mask:
        feature_map[tfexample_fields.sem_seg] = (
            tf.FixedLenFeature([], dtype=tf.string, default_value=''))
    
    features = tf.parse_single_example(serialized=serialized_example, features=feature_map)

    image_decoded = tf.image.decode_image(
        features[tfexample_fields.image_encoded], channels=3)
    image_decoded = tf.cast(image_decoded, tf.float32)
    image_decoded.set_shape([None, None, 3])

    height = tf.cast(features[tfexample_fields.height], tf.int32)
    width = tf.cast(features[tfexample_fields.width], tf.int32)
    orig_shape = tf.stack([height, width], axis=0)

    filename = features[tfexample_fields.filename]
    key = features[tfexample_fields.key]

    ymin = tf.expand_dims(tf.sparse_tensor_to_dense(
        features[tfexample_fields.object_bbox_ymin]), axis=1)
    xmin = tf.expand_dims(tf.sparse_tensor_to_dense(
        features[tfexample_fields.object_bbox_xmin]), axis=1)
    ymax = tf.expand_dims(tf.sparse_tensor_to_dense(
        features[tfexample_fields.object_bbox_ymax]), axis=1)
    xmax = tf.expand_dims(tf.sparse_tensor_to_dense(
        features[tfexample_fields.object_bbox_xmax]), axis=1)
    gt_boxes = tf.concat([ymin, xmin, ymax, xmax], 1)

    gt_classes = tf.sparse_tensor_to_dense(features[tfexample_fields.object_class_label])
    gt_is_crowd = tf.sparse_tensor_to_dense(features[tfexample_fields.object_is_crowd])
    gt_difficult = tf.sparse_tensor_to_dense(features[tfexample_fields.object_difficult])

    def decode_mask(mask_encoded):
        mask_decode = tf.squeeze(
            tf.image.decode_image(mask_encoded, channels=1), axis=2)
        mask_decode.set_shape([None, None])
        return tf.cast(tf.round(mask_decode), tf.float32)

    tensor_dict = {
        input_fields.image: image_decoded,
        input_fields.filename: filename,
        input_fields.key: key,
        input_fields.orig_shape: orig_shape,
        input_fields.gt_boxes: gt_boxes,
        input_fields.gt_classes: gt_classes,
        input_fields.gt_is_crowd: gt_is_crowd,
        input_fields.gt_difficult: gt_difficult,
    }
    if load_instance_masks:
        instance_masks = tf.sparse_tensor_to_dense(
            features[tfexample_fields.instance_masks], default_value='')
        gt_masks = tf.cond(
            tf.greater(tf.size(instance_masks), 0),
            lambda: tf.map_fn(decode_mask, instance_masks, dtype=tf.float32),
            lambda: tf.zeros(tf.cast(tf.stack([0, height, width]), tf.int32)))
        tensor_dict[input_fields.gt_masks] = gt_masks

    if load_semantic_mask:
        sem_seg = tf.image.decode_image(
            features[tfexample_fields.sem_seg], channels=3
        )
        sem_seg.set_shape([None, None, None])
        tensor_dict[input_fields.sem_seg] = id_utils.rgb2id(sem_seg)
    return tensor_dict
