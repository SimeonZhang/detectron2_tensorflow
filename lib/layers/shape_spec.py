from collections import namedtuple
from abc import ABCMeta, abstractmethod
import tensorflow as tf

__all__ = ["ShapeSpec"]


class ShapeSpec(namedtuple(
        "_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to obtain the shape inference ability among pytorch modules.
    Attributes:
        channels:
        height:
        width:
        stride:
    """

    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)
