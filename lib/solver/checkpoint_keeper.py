import os
import glob
import tensorflow as tf


class CheckpointKeepingHook(tf.train.SessionRunHook):

    def __init__(self, cfg, checkpoint_dir, saver=None, global_step=None):
        if cfg.SOLVER.SHORT_TERM_NUM_STEPS % cfg.SOLVER.SHORT_TERM_SAVE_STEPS != 0:
            raise ValueError(
                "`SHORT_TERM_NUM_STEPS` is not divisible by `SHORT_TERM_SAVE_STEPS`."
            )
        if cfg.SOLVER.LONG_TERM_SAVE_STEPS % cfg.SOLVER.SHORT_TERM_SAVE_STEPS != 0:
            raise ValueError(
                "`LONG_TERM_SAVE_STEPS` is not divisible by `SHORT_TERM_SAVE_STEPS`."
            )
        self.short_term_num_steps = cfg.SOLVER.SHORT_TERM_NUM_STEPS
        self.short_term_save_steps = cfg.SOLVER.SHORT_TERM_SAVE_STEPS
        self.long_term_save_steps = cfg.SOLVER.LONG_TERM_SAVE_STEPS
        self.checkpoint_dir = checkpoint_dir
        self.saver = saver
        self.global_step = global_step

    def begin(self):
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=100000)
        if self.global_step is None:
            self.global_step = tf.train.get_or_create_global_step()

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(fetches=self.global_step)

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        if global_step % self.short_term_save_steps == 0 or run_context.stop_requested:
            # save checkpoint
            tf.logging.info(f"global step {global_step}: saving model to disk.")
            checkpoint_path_to_save = os.path.join(
                self.checkpoint_dir, "model-{:d}".format(global_step)
            )
            self.saver.save(run_context.session, checkpoint_path_to_save)

            if global_step > self.short_term_num_steps:
                latest_step_to_delete = global_step - self.short_term_num_steps
                if latest_step_to_delete % self.long_term_save_steps != 0:
                    checkpoint_path_to_delete = os.path.join(
                        self.checkpoint_dir, "model-{:d}*".format(latest_step_to_delete)
                    )
                    for fid in glob.glob(checkpoint_path_to_delete):
                        os.remove(fid)
