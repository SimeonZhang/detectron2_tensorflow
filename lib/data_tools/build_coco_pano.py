r"""Convert raw COCO dataset to TFRecord for panoptic segmentation.
Please note that this tool creates sharded output files.
Example usage:
    python build_coco_pano.py --logtostderr \
      --root_dir="${ROOT_DIR}" \
      --output_dir="${OUTPUT_DIR}"
"""
import io
import json
import os
import shutil
import contextlib2

import numpy as np
from PIL import Image
import tensorflow as tf
from pycocotools import mask
from ..panopticapi.utils import rgb2id, id2rgb

from ..data import fields
from ..utils import tfrecord_creation_utils
from ..utils import dataset_utils


flags = tf.app.flags
tf.flags.DEFINE_string('root_dir', '', 'root dir containing coco images and annotations.')
tf.flags.DEFINE_string('output_dir', '', 'Output data directory.')
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def create_tf_example(image,
                      annotation,
                      image_dir,
                      panoptic_dir,
                      category_map):
    """Converts image and annotations to a tf.Example proto.
    Args:
      image: dict with keys:
        [u'license', u'file_name', u'coco_url', u'height', u'width',
        u'date_captured', u'flickr_url', u'id']
      annotation:
        list of dicts with keys:
        [u'area', u'iscrowd', u'image_id',  u'bbox', u'category_id', u'id']
        Notice that bounding box coordinates in the official COCO dataset are
        given as [x, y, width, height] tuples using absolute coordinates where
        x, y represent the top-left (0-indexed) corner.  This function converts
        to the format of [ymin, xmin, ymax, xmax] with coordinates normalized
        relative to image size).
      image_dir: directory containing the image files.
      panoptic_dir:
      category_map
    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.
    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']

    full_path = os.path.join(image_dir, filename)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    key = filename.split(".")[0]

    panoptic_path = os.path.join(panoptic_dir, annotation['file_name'])
    panoptic = np.asarray(Image.open(panoptic_path), dtype=np.uint32)
    panoptic = rgb2id(panoptic)

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    object_is_crowd = []
    object_difficult = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    sem_seg = np.zeros_like(panoptic, dtype=np.uint64) + category_map["stuff_ignore_value"]
    num_annotations_skipped = 0
    for segment_info in annotation['segments_info']:
        category_id = int(segment_info['category_id'])
        if category_id in category_map['thing_id_map']:
            x, y, width, height = tuple(segment_info['bbox'])
            if width <= 0 or height <= 0:
                num_annotations_skipped += 1
                continue
            xmin.append(float(x) / image_width)
            xmax.append(float(x + width) / image_width)
            ymin.append(float(y) / image_height)
            ymax.append(float(y + height) / image_height)

            object_is_crowd.append(segment_info['iscrowd'])
            object_difficult.append(0)

            continuous_category_id = category_map['thing_id_map'][category_id]
            category_ids.append(continuous_category_id)
            category_names.append(
                category_map['category_index'][category_id]['name'].encode('utf8'))
            area.append(segment_info['area'])

            binary_mask = np.uint8(panoptic == segment_info['id'])
            pil_mask = Image.fromarray(binary_mask)
            output_io = io.BytesIO()
            pil_mask.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())
        continuous_category_id = category_map['stuff_id_map'][category_id]
        sem_seg[panoptic == segment_info["id"]] = continuous_category_id
    output_io = io.BytesIO()
    Image.fromarray(id2rgb(sem_seg), mode="RGB").save(output_io, format='PNG')
    encoded_sem_seg = output_io.getvalue()

    feature_dict = {
        fields.TfExampleFields.height:
            dataset_utils.int64_feature(image_height),
        fields.TfExampleFields.width:
            dataset_utils.int64_feature(image_width),
        fields.TfExampleFields.filename:
            dataset_utils.bytes_feature(filename.encode('utf8')),
        fields.TfExampleFields.key:
            dataset_utils.bytes_feature(key.encode('utf8')),
        fields.TfExampleFields.image_encoded:
            dataset_utils.bytes_feature(encoded_jpg),
        fields.TfExampleFields.image_format:
            dataset_utils.bytes_feature('jpeg'.encode('utf8')),
        fields.TfExampleFields.object_bbox_xmin:
            dataset_utils.float_list_feature(xmin),
        fields.TfExampleFields.object_bbox_xmax:
            dataset_utils.float_list_feature(xmax),
        fields.TfExampleFields.object_bbox_ymin:
            dataset_utils.float_list_feature(ymin),
        fields.TfExampleFields.object_bbox_ymax:
            dataset_utils.float_list_feature(ymax),
        fields.TfExampleFields.object_class_text:
            dataset_utils.bytes_list_feature(category_names),
        fields.TfExampleFields.object_class_label:
            dataset_utils.int64_list_feature(category_ids),
        fields.TfExampleFields.object_is_crowd:
            dataset_utils.int64_list_feature(object_is_crowd),
        fields.TfExampleFields.object_difficult:
            dataset_utils.int64_list_feature(object_difficult),
        fields.TfExampleFields.instance_masks:
            dataset_utils.bytes_list_feature(encoded_mask_png),
        fields.TfExampleFields.sem_seg:
            dataset_utils.bytes_feature(encoded_sem_seg),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, num_annotations_skipped


def _create_tf_record_from_coco_annotations(
        annotations_file, image_dir, panoptic_dir, output_path, split_name, num_shards):
    """Loads COCO annotation json files and converts to tf.Record format.
    Args:
      annotations_file: JSON file containing bounding box annotations.
      image_dir: Directory containing the image files.
      panoptic_dir: Directory containing the panoptic png annotations.
      output_path: Path to output tf.Record file.
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
      num_shards: number of output file shards.
    """
    base_output_path = os.path.join(output_path, split_name)
    with contextlib2.ExitStack() as tf_record_close_stack, \
            tf.gfile.GFile(annotations_file, 'r') as fid:
        output_tfrecords = tfrecord_creation_utils.open_sharded_output_tfrecords(
            tf_record_close_stack, base_output_path, num_shards)
        groundtruth_data = json.load(fid)
        images = groundtruth_data['images']
        categories = groundtruth_data['categories']
        stuff_ids = [k["id"] for k in categories if k["isthing"] == 0]
        thing_ids = [k["id"] for k in categories if k["isthing"] == 1]
        stuff_id_map = {}
        thing_id_map = {}
        stuff_ignore_value = len(stuff_ids) + 1
        for i, thing_id in enumerate(thing_ids):
            thing_id_map[thing_id] = i
            stuff_id_map[thing_id] = 0
        for j, stuff_id in enumerate(stuff_ids):
            stuff_id_map[stuff_id] = j + 1
        stuff_id_map[0] = stuff_ignore_value

        num_thing_classes = len(thing_ids)
        num_stuff_classes = len(stuff_ids) + 1
        category_index = {item['id']: item for item in categories}
        category_map = {
            'category_index': category_index,
            'thing_id_map': thing_id_map,
            'stuff_id_map': stuff_id_map,
            "num_thing_classes": num_thing_classes,
            "num_stuff_classes": num_stuff_classes,
            "stuff_ignore_value": stuff_ignore_value
        }
        category_map_path = os.path.join(output_path, 'category_map.json')
        with open(category_map_path, 'w') as fid:
            json.dump(category_map, fid, indent=4)

        annotation_index = {}
        if 'annotations' in groundtruth_data:
            tf.logging.info(
                'Found groundtruth annotations. Building annotations index.')
            for annotation in groundtruth_data['annotations']:
                annotation_index[annotation['image_id']] = annotation
        missing_annotation_count = 0
        filtered_image_ids = []
        for image in images:
            image_id = image['id']
            if image_id not in annotation_index:
                missing_annotation_count += 1
                filtered_image_ids.append(image_id)
                # annotations_index[image_id] = []
        tf.logging.info('%d images are missing annotations.', missing_annotation_count)

        total_num_annotations_skipped = 0
        for idx, image in enumerate(images):
            if idx % 100 == 0:
                tf.logging.info('On image %d of %d', idx, len(images))
            if image['id'] in filtered_image_ids:
                continue
            annotation = annotation_index[image['id']]
            tf_example, num_annotations_skipped = create_tf_example(
                image, annotation, image_dir, panoptic_dir, category_map)
            total_num_annotations_skipped += num_annotations_skipped
            shard_idx = idx % num_shards
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())
        tf.logging.info('Finished writing, skipped %d annotations.', total_num_annotations_skipped)


def build(cfg):
    if os.path.exists(cfg.DATASETS.ROOT_DIR):
        shutil.rmtree(cfg.DATASETS.ROOT_DIR)
    os.makedirs(cfg.DATASETS.ROOT_DIR)

    train_image_dir = os.path.join(cfg.BUILD_RECORDS.ROOT_DIR, 'train2017')
    train_annotations_file = os.path.join(
        cfg.BUILD_RECORDS.ROOT_DIR, 'annotations', 'panoptic_train2017.json')
    train_panoptic_dir = os.path.join(cfg.BUILD_RECORDS.ROOT_DIR, 'panoptic_train2017')
    val_image_dir = os.path.join(cfg.BUILD_RECORDS.ROOT_DIR, 'val2017')
    val_annotations_file = os.path.join(
        cfg.BUILD_RECORDS.ROOT_DIR, 'annotations', 'panoptic_val2017.json')
    val_panoptic_dir = os.path.join(cfg.BUILD_RECORDS.ROOT_DIR, 'panoptic_val2017')

    _create_tf_record_from_coco_annotations(
        train_annotations_file,
        train_image_dir,
        train_panoptic_dir,
        cfg.DATASETS.ROOT_DIR,
        cfg.DATASETS.TRAIN,
        num_shards=cfg.BUILD_RECORDS.TRAIN_NUM_SHARDS)
    _create_tf_record_from_coco_annotations(
        val_annotations_file,
        val_image_dir,
        val_panoptic_dir,
        cfg.DATASETS.ROOT_DIR,
        cfg.DATASETS.VAL,
        num_shards=cfg.BUILD_RECORDS.VAL_NUM_SHARDS)

    target_panoptic_dir = os.path.join(cfg.DATASETS.ROOT_DIR, f"panoptic_{cfg.DATASETS.VAL}")
    os.symlink(val_panoptic_dir, target_panoptic_dir)
    target_panoptic_file = os.path.join(cfg.DATASETS.ROOT_DIR, f"panoptic_{cfg.DATASETS.VAL}.json")
    os.symlink(val_annotations_file, target_panoptic_file)
