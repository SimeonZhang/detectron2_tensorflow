import tensorflow as tf


def build_learning_rate(cfg):
    global_step = tf.train.get_or_create_global_step()
    global_step = tf.cast(global_step, tf.float32)

    boundaries = [float(x) for x in cfg.SOLVER.STEPS]
    values = []
    for i in range(len(boundaries) + 1):
        values.append(cfg.SOLVER.BASE_LR * cfg.SOLVER.GAMMA ** i)
    if cfg.SOLVER.AUTO_SCALE_LR_SCHEDULE:
        factor = cfg.SOLVER.IMS_PER_BATCH / cfg.SOLVER.IMS_PER_BATCH_BASE
        boundaries = [x / factor for x in boundaries]
        values = [x * factor for x in values]
    learning_rate = tf.train.piecewise_constant(
        global_step, boundaries, values
    )

    warmup_factor = _get_warmup_factor_at_step(
        global_step, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_FACTOR
    )
    learning_rate = learning_rate * warmup_factor
    return learning_rate


def _get_warmup_factor_at_step(
    global_step,
    warmup_iters,
    warmup_factor
):
    def true_fn():
        alpha = global_step / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    warmup_factor = tf.cond(
        global_step < warmup_iters, true_fn, lambda: 1.0
    )
    return warmup_factor
