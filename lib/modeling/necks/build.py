from typing import Dict

from ...layers import ShapeSpec
from ...utils.registry import Registry
from .neck import Neck

NECK_REGISTRY = Registry("NECK")
NECK_REGISTRY.__doc__ = """
Registry for necks, which extract feature maps from images
The registered object must be a callable that accepts two arguments:
1. A :class:`cocktail.config.CfgNode`
2. A :class:`cocktail.layers.ShapeSpec`,
   which contains the input shape specification.
It must returns an instance of :class:`Backbone`.
"""


@NECK_REGISTRY.register()
class DummyNeck(Neck):

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec], **kwargs):
        """
        Necks fuse or perform additional operations on the original features
        from backbone.
        """
        super(DummyNeck, self).__init__(**kwargs)
        self._out_feature_strides = {k: v.stride for k, v in input_shape.items()}
        self._out_feature_channels = {k: v.channels for k, v in input_shape.items()}
        self._out_features = list(sorted(input_shape.keys()))

    def call(self, features):
        """
        Args:
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are height, width, and channels, which may
                vary between feature maps (e.g., if a feature pyramid is used).
        Returns:
            dict[str: Tensor]: mapping from feature name (e.g., "p2")
             to tensor
        """
        return features

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
        # this is a backward-compatible default
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


def build_neck(cfg, input_shape, **kwargs):
    """
    Build a neck from `cfg.MODEL.NECK.NAME`.
    Returns:
        an instance of :class:`NECK`
    """
    name = cfg.MODEL.NECK.NAME
    if name == "":
        neck = DummyNeck(cfg, input_shape, **kwargs)
    else:
        neck = NECK_REGISTRY.get(name)(cfg, input_shape, **kwargs)
    assert isinstance(neck, Neck)
    return neck