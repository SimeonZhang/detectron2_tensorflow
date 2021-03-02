from ...layers import ShapeSpec
from ...utils.registry import Registry

from .backbone import Backbone

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images
The registered object must be a callable that accepts two arguments:
1. A :class:`cocktail.config.CfgNode`
2. A :class:`cocktail.layers.ShapeSpec`,
   which contains the input shape specification.
It must returns an instance of :class:`Backbone`.
"""


def build_backbone(cfg, input_shape=None, **kwargs):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape, **kwargs)
    assert isinstance(backbone, Backbone)
    return backbone
