# @Author: zhangxiangwei
# @Date: 2020-02-26 17:21:51
# @Last Modified by:   zhangxiangwei
# @Last Modified time: 2020-02-26 17:21:51
import os
import re
import json
from abc import ABCMeta, abstractmethod

import tensorflow as tf
from tensorflow.python.tools import freeze_graph

from ..utils import registry


EXPORTER_REGISTRY = registry.Registry("exporter")
EXPORTER_REGISTRY.__doc__ = """
Registry for exporters, which build frozen graph and write saved models
The registered object must be a callable that accepts a argument:
   A :class:`cocktail.config.CfgNode`
   which contains the input shape specification.
It must returns an instance of :class:`Exporter`.
"""


def export(cfg):
    return EXPORTER_REGISTRY.get(cfg.SERVING_MODEL.TYPE)()(cfg)


class Exporter(object, metaclass=ABCMeta):
    """
    Abstract base class for exporters.
    """

    def __call__(self, cfg):
        self.label_offset = cfg.SERVING_MODEL.LABEL_OFFSET
        self.tensor_prefix = cfg.SERVING_MODEL.INPUT_OUTPUT_TENSOR_PREFIX
        output_dir = os.path.join(cfg.LOGS.ROOT_DIR, cfg.LOGS.EXPORT)
        if not os.path.exists(output_dir):
            version = 0
        else:
            old_version = 0
            for v in os.listdir(output_dir):
                if re.match("^\d+$", v) and int(v) > old_version:
                    old_version = int(v)
            version = old_version + 1
        output_dir = os.path.join(output_dir, str(version))

        checkpoint_dir = os.path.join(cfg.LOGS.ROOT_DIR, cfg.LOGS.TRAIN)
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        categoty_map_path = os.path.join(
            cfg.DATASETS.ROOT_DIR, cfg.DATASETS.CATEGORY_MAP_NAME
        )
        with open(categoty_map_path) as fid:
            self.category_map = json.load(fid)

        inputs, outputs = self.build_inference_graph(cfg)
        
        saver = tf.train.Saver()
        input_saver_def = saver.as_saver_def()

        frozen_graph_def = freeze_graph.freeze_graph_with_def_protos(
            input_graph_def=tf.get_default_graph().as_graph_def(),
            input_saver_def=input_saver_def,
            input_checkpoint=checkpoint,
            output_node_names=",".join(outputs.keys()),
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=None,
            clear_devices=True,
            initializer_nodes='',
        )

        self.write_saved_model(output_dir, frozen_graph_def, inputs, outputs)

        frozen_graph_path = os.path.join(
            output_dir, cfg.SERVING_MODEL.FROZEN_GRAPH_FILE_NAME
        )
        with open(frozen_graph_path, 'wb') as fid:
            fid.write(frozen_graph_def.SerializeToString())

        label_index_map_path = os.path.join(output_dir, "label_index.map")
        label_index_map = []
        for _, category in self.category_map["category_index"].items():
            label_index_map.append(
                {
                    "id": category["id"] + self.label_offset,
                    "class": category["name"]
                }
            )
        with open(label_index_map_path, "w") as fid:
            json.dump(label_index_map, fid, indent=2)

        model_info_path = os.path.join(output_dir, "model_info.json")
        with open(model_info_path, "w") as fid:
            model_info = {
                "input_output_tensor_prefix": cfg.SERVING_MODEL.INPUT_OUTPUT_TENSOR_PREFIX
            }
            json.dump(model_info, fid, indent=4, sort_keys=True)


    def write_saved_model(self, saved_model_path, frozen_graph_def, inputs, outputs):
        """write saved model."""
        with tf.Graph().as_default():
            with tf.Session() as sess:

                tf.import_graph_def(frozen_graph_def, name='')
                builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

                signature_def_map = self.build_signature_def_map(inputs, outputs)
                builder.add_meta_graph_and_variables(
                    sess,
                    [tf.saved_model.tag_constants.SERVING],
                    signature_def_map=signature_def_map
                )
                builder.save()

    @abstractmethod
    def build_inference_graph(self):
        """build the inference graph."""
        pass

    @abstractmethod
    def build_signature_def_map(self, inputs, outputs):
        """build signature def map to write saved model."""
        pass
