import tensorflow as tf

import logging
from absl import flags

from lib.config import get_cfg
from lib.convert_models import save_checkpoint
from lib.utils import config_utils

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
    config_utils.finalize(cfg, training=False)
    save_checkpoint.save(cfg)

if __name__ == '__main__':
    tf.app.run()
