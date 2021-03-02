import os
import tensorflow as tf

from tensorflow.python.ops import variables as tf_variables

from ..data import fields
from ..data.dataloader import build_dataloader
from ..modeling.meta_arch import build_model
from . import model_deploy
from ..structures import box_list
from ..solver.learning_rate import build_learning_rate
from ..solver.regularizer import get_regularizer_losses
from ..solver.checkpoint_keeper import CheckpointKeepingHook

slim = tf.contrib.slim


def get_batched_inputs(cfg):
    input_fields = fields.InputFields
    dataloader = build_dataloader(cfg, training=True)
    tensor_dict = dataloader.get_next()

    batched_inputs = {
        'image': tensor_dict[input_fields.image],
        'image_shape': tensor_dict[input_fields.true_shape],
    }

    if input_fields.sem_seg in tensor_dict:
        batched_inputs['sem_seg'] = tensor_dict[input_fields.sem_seg]

    instances = box_list.BoxList(tensor_dict[input_fields.gt_boxes])
    instances.add_field('gt_classes', tensor_dict[input_fields.gt_classes])
    instances.add_field('gt_difficult', tensor_dict[input_fields.gt_difficult])
    instances.add_field('gt_is_crowd', tensor_dict[input_fields.gt_is_crowd])
    instances.add_field('is_valid', tensor_dict[input_fields.is_valid])
    if input_fields.gt_masks in tensor_dict:
        instances.add_field('gt_masks', tensor_dict[input_fields.gt_masks])
    batched_inputs['instances'] = instances

    return batched_inputs


def train(cfg):
    checkpoint_dir = os.path.join(cfg.LOGS.ROOT_DIR, cfg.LOGS.TRAIN)

    with tf.Graph().as_default():

        # Build a configuration specifying multi-GPU and multi-replicas.
        num_clones = cfg.SOLVER.NUM_GPUS
        deploy_config = model_deploy.DeploymentConfig(num_clones)

        with tf.device(deploy_config.inputs_device()):
            batched_inputs = get_batched_inputs(cfg)

        # Place the global step on the device storing the variables.
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

            with slim.arg_scope(
                    [slim.model_variable, slim.variable], device=deploy_config.variables_device()
            ):
                model_fn = build_model(cfg)
                clones = model_deploy.create_clones(deploy_config, model_fn, [batched_inputs])

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        global_summaries = set([])

        init_fn = None
        if cfg.PRETRAINS.WEIGHTS:
            checkpoint_path = os.path.join(cfg.PRETRAINS.ROOT, cfg.PRETRAINS.WEIGHTS)
            variable_names_map = {}
            for variable in slim.get_variables_to_restore():
                if isinstance(variable, tf_variables.PartitionedVariable):
                    name = variable.name
                else:
                    name = variable.op.name
                if cfg.PRETRAINS.ONLY_BACKBONE:
                    if cfg.MODEL.ROI_HEADS.NAME == "Res5ROIHeads" and "res5" in name:
                        name = name.replace("roi_heads", "backbone")
                variable_names_map[name] = variable

            ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
            ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
            ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)

            vars_in_ckpt = {}
            for variable_name, variable in sorted(variable_names_map.items()):
                if variable_name in ckpt_vars_to_shape_map:
                    if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
                        vars_in_ckpt[variable_name] = variable
                    else:
                        tf.logging.warning(
                            'Variable [%s] is available in checkpoint, but has an '
                            'incompatible shape with model variable. Checkpoint '
                            'shape: [%s], model variable shape: [%s]. This '
                            'variable will not be initialized from the checkpoint.',
                            variable_name, ckpt_vars_to_shape_map[variable_name],
                            variable.shape.as_list())
                else:
                    tf.logging.warning(
                        'Variable [%s] is not available in checkpoint', variable_name)
            init_saver = tf.train.Saver(vars_in_ckpt)

            def initializer_fn(scaffold, sess):
                init_saver.restore(sess, checkpoint_path)
            init_fn = initializer_fn

        first_clone_scope = clones[0].scope
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # ------------------------------------------------------
        # Configure the optimization procedure.
        # ------------------------------------------------------
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = build_learning_rate(cfg)
            global_summaries.add(tf.summary.scalar('learning_rate', learning_rate))
            optimizer = tf.train.MomentumOptimizer(learning_rate, cfg.SOLVER.MOMENTUM)

            regularization_losses = get_regularizer_losses(cfg)
            total_loss, grads_and_vars = model_deploy.optimize_clones(
                clones, optimizer, regularization_losses=regularization_losses
            )
            # total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')
            global_summaries.add(tf.summary.scalar('total_loss', total_loss))

            if cfg.SOLVER.CLIP_GRADIENTS_BY_NORM > 0:
                with tf.name_scope("ClipGradients"):
                    grads_and_vars = slim.learning.clip_gradient_norms(
                        grads_and_vars, cfg.SOLVER.CLIP_GRADIENTS_BY_NORM)

            # Create gradient updates.
            grad_updates = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)
            update_ops.append(grad_updates)

            update_op = tf.group(*update_ops)
            with tf.control_dependencies([update_op]):
                train_op = tf.identity(total_loss, name='train_op')

        # -------------------------------------------------------
        # add summaries
        # -------------------------------------------------------
        for model_var in slim.get_model_variables():
            global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))
        summaries |= global_summaries

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # Soft placement allows placing on CPU ops without GPU implementation.
        session_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False
        )
        number_of_steps = cfg.SOLVER.MAX_ITER
        if cfg.SOLVER.AUTO_SCALE_LR_SCHEDULE:
            factor = cfg.SOLVER.IMS_PER_BATCH / cfg.SOLVER.IMS_PER_BATCH_BASE
            number_of_steps = int(round(number_of_steps / factor))

        saver = tf.train.Saver(max_to_keep=100000)
        global_step = tf.train.get_or_create_global_step()
        hooks = [
            CheckpointKeepingHook(cfg, checkpoint_dir, saver, global_step)
        ]
        scaffold = tf.train.Scaffold(init_fn=init_fn)

        def step_fn(step_context):
            global_step_np = step_context.session.run(global_step)

            if global_step_np == 0:
                # save initialized checkpoint
                tf.logging.info(f"global step 0: saving model to disk.")
                checkpoint_path_to_save = os.path.join(checkpoint_dir, "model-0")
                saver.save(step_context.session, checkpoint_path_to_save)

            if global_step_np >= number_of_steps:
                step_context.request_stop()
            loss = step_context.run_with_hooks(train_op)
            if (global_step_np + 1) % 10 == 0:
                tf.logging.info(
                    "global step {:d}: loss = {:.3f}".format(global_step_np + 1, loss)
                )
            return loss

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=checkpoint_dir,
                scaffold=scaffold,
                chief_only_hooks=hooks,
                save_checkpoint_secs=None,
                save_checkpoint_steps=None,
                config=session_config) as session:
            while not session.should_stop():
                loss = session.run_step_fn(step_fn)
