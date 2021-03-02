from abc import ABCMeta, abstractmethod

from ...layers import Layer, ShapeSpec

__all__ = ["Backbone"]


class Backbone(Layer, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self, **kwargs):
        """
        The `__init__` method of any subclass can specify its own set of
        arguments.
        """
        super().__init__(**kwargs)

    @abstractmethod
    def call(self):
        """
        Subclasses must override this method, but adhere to the same return
        type.
        Returns:
            dict[str: Tensor]: mapping from feature name (e.g., "C2")
             to tensor
        """
        pass

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def out_features(self):
        return self._out_features

