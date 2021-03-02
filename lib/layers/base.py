from abc import ABCMeta, abstractmethod
import tensorflow as tf

from ..utils import tf_utils

slim = tf.contrib.slim

__all__ = ["Layer"]


class Layer(object, metaclass=ABCMeta):

    def __init__(self, dtype=tf.float32, scope=None, **kwargs):
        self.dtype = dtype
        self.scope = self._set_scope(scope)
        for name, value in kwargs.items():
            setattr(self, name, value)
        if not hasattr(self, 'training'):
            self.training = tf_utils.get_training_phase()

    def _set_scope(self, scope=None):
        return self.__class__.__name__ if scope is None else scope

    def build_variable_getter(self, rename=None):
        """Build a model variable getter that respects scope
         getter and renames."""

        # VariableScope will nest the getters
        def layer_variable_getter(getter, *args, **kwargs):
            kwargs['rename'] = rename
            return _model_variable_getter(getter, *args, **kwargs)

        return layer_variable_getter

    def __call__(self, *args, **kwargs):
        with tf.name_scope(self.scope):
            return self.call(*args, **kwargs)

    @abstractmethod
    def call(self):
        raise NotImplementedError


class Sequential(object):

    def __init__(self, layers=None):
        if layers is None:
            self._layers = []
        else:
            assert isinstance(layers, (list, tuple))
            for l in layers:
                assert isinstance(l, Layer)
            self._layers = list(layers)

    def add(self, layer):
        assert isinstance(layer, Layer)
        self._layers.append(layer)

    def __call__(self, inputs):
        ret = inputs
        for layer in self._layers:
            ret = layer(ret)
        return ret


def _model_variable_getter(
        getter,
        name,
        shape=None,
        dtype=None,
        initializer=None,
        regularizer=None,
        trainable=True,
        collections=None,
        caching_device=None,
        partitioner=None,
        rename=None,
        use_resource=None,
        synchronization=tf.VariableSynchronization.AUTO,
        aggregation=tf.VariableAggregation.NONE,
        **_):
    """Getter that uses model_variable for compatibility with core layers."""
    short_name = name.split('/')[-1]
    if rename and short_name in rename:
        name_components = name.split('/')
        name_components[-1] = rename[short_name]
        name = '/'.join(name_components)
    return slim.model_variable(
        name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        collections=collections,
        trainable=trainable,
        caching_device=caching_device,
        partitioner=partitioner,
        custom_getter=getter,
        use_resource=use_resource,
        synchronization=synchronization,
        aggregation=aggregation)
