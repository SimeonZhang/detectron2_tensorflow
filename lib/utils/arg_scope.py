import copy
from contextlib import contextmanager
from functools import wraps
from collections import defaultdict
import tensorflow as tf

_ArgScopeStack = []


@contextmanager
def arg_scope(layers, **kwargs):
    """
    Args:
        layers (list or layer): layer or list of layers to apply the arguments.

    Returns:
        a context where all appearance of these layer will by default have the
        arguments specified by kwargs.

    Example:
        .. code-block:: python

            with arg_scope(Conv2D, kernel_shape=3, nl=tf.nn.relu, out_channel=32):
                x = Conv2D('conv0', x)
                x = Conv2D('conv1', x)
                x = Conv2D('conv2', x, out_channel=64)  # override argscope

    """
    if not isinstance(layers, list):
        layers = [layers]

    for l in layers:
        assert hasattr(l, '__arg_scope_enabled__'), "Argscope not supported for {}".format(l)

    # need to deepcopy so that changes to new_scope does not affect outer scope
    new_scope = copy.deepcopy(get_arg_scope())
    for l in layers:
        new_scope[l.__name__].update(kwargs)
    _ArgScopeStack.append(new_scope)
    yield
    del _ArgScopeStack[-1]


def get_arg_scope():
    """
    Returns:
        dict: the current argscope.

    An argscope is a dict of dict: ``dict[layername] = {arg: val}``
    """
    if len(_ArgScopeStack) > 0:
        return _ArgScopeStack[-1]
    else:
        return defaultdict(dict)


def add_arg_scope(cls):
    """Decorator for function to support argscope

    Example:

        .. code-block:: python

            from mylib import MyClass
            myfunc = add_arg_scope(MyClass)

    Args:
        func: A function mapping one or multiple tensors to one or multiple
            tensors.
    Remarks:
        If the function ``func`` returns multiple input or output tensors,
        only the first input/output tensor shape is displayed during logging.

    Returns:
        The decorated function.

    """
    original_init = cls.__init__

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        actual_args = copy.copy(get_arg_scope()[cls.__name__])
        actual_args.update(kwargs)
        instance = original_init(self, *args, **actual_args)
        return instance
    cls.__arg_scope_enabled__ = True
    cls.__init__ = wrapped_init
    return cls
