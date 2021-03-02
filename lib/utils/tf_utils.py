import tensorflow as tf
from tensorflow.python.client import device_lib


_TRAINING = None


def get_training_phase():
    if _TRAINING is None:
        raise ValueError('you must set training phase!')
    return _TRAINING


def set_training_phase(training: bool):
    global _TRAINING
    _TRAINING = training


def get_tf_version_tuple():
    """
    Return TensorFlow version as a 2-element tuple (for comparison).
    """
    return tuple(map(int, tf.__version__.split('.')[:2]))


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
