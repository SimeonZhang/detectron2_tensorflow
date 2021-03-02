"""Utility functions for creating TFRecord data sets."""

from lxml import etree
import tensorflow as tf
import os
import json
import requests

GET_NAME_URL = "http://label-api.ainnovation.com:8080/api/label/name"


def get_sku_info(uuid, retry=3):
    if retry <= 0:
        return None
    try:
        params = {"uuid": uuid}
        res = requests.get(GET_NAME_URL, params=params, timeout=30)
        return res.json().get("data", {}).get("name", {})
    except requests.RequestException as e:
        print("RequestException: %s" % str(e))
        return _request_sku_name(uuid, retry - 1)


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.

    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
      Python dictionary holding XML contents.
    """
    if xml is None:
        return {}
    if len(xml) == 0:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = _recursive_parse_xml_to_dict(child)
        if child.tag not in ['object', 'point']:
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def read_annotation(annotation_path):
    with open(annotation_path, 'rb') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = _recursive_parse_xml_to_dict(xml)["annotation"]
    return data


def read_examples_list(path):
    """Read list of training or validation examples.
    The file is assumed to contain a single example per line where the first
    token in the line is an identifier that allows us to find the image and
    annotation xml for that example.
    For example, the line:
    xyz 3
    would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).
    Args:
      path: absolute path to examples list file.
    Returns:
      list of example identifiers (strings).
    """
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]
