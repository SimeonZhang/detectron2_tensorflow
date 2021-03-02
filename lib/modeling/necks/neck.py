from abc import ABCMeta, abstractmethod

from ...layers import Layer, ShapeSpec

__all__ = ["Neck"]


class Neck(Layer, metaclass=ABCMeta):
    """
    Abstract base class for necks.
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
            dict[str: Tensor]: mapping from feature name (e.g., "P2")
             to tensor
        """
        pass

    @property
    def size_divisibility(self):
        """
        Some necks require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type
        networks with lateral connection (e.g., FPN) for which feature maps
        need to match dimension in the "bottom up" and "top down" paths.
        Set to 0 if no specific input size divisibility is required.
        """
        return 0

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

