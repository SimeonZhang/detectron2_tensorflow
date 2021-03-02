from ...utils.registry import Registry

SINGLE_STAGE_HEADS_REGISTRY = Registry("SINGLE_STAGE_HEADS")
SINGLE_STAGE_HEADS_REGISTRY.__doc__ = """
Registry for heads in a single stage detector, which take feature maps and give
detection or instance segmentation results.
The registered object will be called with `obj(cfg, input_shape)`.
"""


def build_single_stage_head(cfg, input_shape, **kwargs):
    """
    Build a single stage head from `cfg.MODEL.SINGLE_STAGE_HEAD.NAME`.
    """
    name = cfg.MODEL.SINGLE_STAGE_HEAD.NAME

    return SINGLE_STAGE_HEADS_REGISTRY.get(name)(cfg, input_shape, **kwargs)


