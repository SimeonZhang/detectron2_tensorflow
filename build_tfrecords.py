r"""Training executable.

This executable is used to train.
A configuration file can be specified by --config_file.
config options can be specified by --opts.

Example usage:
    python build_tfrecords.py \
        --config_file=...
        --opts=FOO,0.5,BAR,1.0
"""
import tensorflow as tf
import logging
from absl import flags

from lib.config import get_cfg
from lib.data_tools import builder

tf.logging.set_verbosity(tf.logging.INFO)
log = logging.getLogger('tensorflow')
log.propagate = False

flags.DEFINE_string('config_file', None, 'Path to the config file.')
flags.DEFINE_string('opts', None, 'A set of config option split by comma, eg:`FOO,0.5,BAR,1.0`.')

FLAGS = flags.FLAGS


def main(_):
    cfg = get_cfg()
    if FLAGS.config_file:
        cfg.merge_from_file(FLAGS.config_file)
    if FLAGS.opts:
        options = FLAGS.opts.split(",")
        cfg.merge_from_list(options)
    cfg.freeze()
    builder.build(cfg)

if __name__ == '__main__':
    tf.app.run()
