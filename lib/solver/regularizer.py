import tensorflow as tf

slim = tf.contrib.slim


def get_regularizer_losses(cfg, scope=None):
    with tf.name_scope(scope, "Regularizer"):
        regularizer_losses = []
        weights_regularizer = slim.l2_regularizer(cfg.SOLVER.WEIGHT_DECAY)
        bias_regularizer = slim.l2_regularizer(cfg.SOLVER.WEIGHT_DECAY_BIAS)
        norm_regularizer = slim.l2_regularizer(cfg.SOLVER.WEIGHT_DECAY_NORM)
        for variable in tf.trainable_variables():
            variable_name = variable.op.name
            if variable_name.endswith('gamma') or variable_name.endswith('beta'):
                regularizer_loss = norm_regularizer(variable)
            elif variable_name.endswith('bias'):
                regularizer_loss = bias_regularizer(variable)
            elif variable_name.endswith('weights'):
                regularizer_loss = weights_regularizer(variable)
            if regularizer_loss is not None:
                tf.logging.info("Add regularizer for %s.", variable_name)
                regularizer_losses.append(regularizer_loss)
        tf.logging.info("Find %s variables to regularize.", len(regularizer_losses))
    return regularizer_losses
