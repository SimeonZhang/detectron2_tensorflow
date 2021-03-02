r"""Evalating executable.

This executable is used to eval.
A configuration file can be specified by --config_file.
config options can be specified by --opts.

Example usage:
    python eval.py \
        --config_file=...
        --opts=FOO,0.5,BAR,1.0
"""
import os
import json
import tensorflow as tf
import logging

from lib.config import get_cfg
from lib.engine import evaluator
from lib.utils import config_utils


tf.logging.set_verbosity(tf.logging.INFO)
log = logging.getLogger('tensorflow')
log.propagate = False

flags = tf.app.flags
flags.DEFINE_string('config_file', '', 'Path to the config file.')
flags.DEFINE_string('opts', '', 'A set of config option split by comma, eg:`FOO,0.5,BAR,1.0`.')

FLAGS = flags.FLAGS


def main(_):
    cfg = get_cfg()
    if FLAGS.config_file:
        cfg.merge_from_file(FLAGS.config_file)
    if FLAGS.opts:
        options = FLAGS.opts.split(",")
        cfg.merge_from_list(options)
    config_utils.finalize(cfg, training=False)
    evaluator.evaluate(cfg)

if __name__ == '__main__':
    tf.app.run()
