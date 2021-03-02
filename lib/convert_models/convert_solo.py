import numpy as np


def convert_weights(d, cfg):
    has_fpn = "FPN" in cfg.MODEL.BACKBONE.NAME
    ret = {}

    def _convert_conv(src, dst):
        src_w = d.pop(src + ".weight").transpose(2, 3, 1, 0)
        ret[dst + "/weights"] = src_w
        if src + ".bias" in d:
            ret[dst + "/bias"] = d.pop(src + ".bias")

    def _convert_norm(src, dst):
        ret[dst + "/norm/gamma"] = d.pop(src + ".weight")
        ret[dst + "/norm/beta"] = d.pop(src + ".bias")
        if src + ".running_var" in d:    # batch norm
            ret[dst + "/norm/moving_variance"] = d.pop(src + ".running_var")
            ret[dst + "/norm/moving_mean"] = d.pop(src + ".running_mean")
            if src + ".num_batches_tracked" in d:
                d.pop(src + ".num_batches_tracked")

    if has_fpn:
        backbone_prefix = "backbone."
        dst_prefix = "backbone/bottom_up/"
    else:
        backbone_prefix = "backbone."
        dst_prefix = "backbone/"
    _convert_conv(backbone_prefix + "conv1", dst_prefix + "stem/conv1")
    _convert_norm(backbone_prefix + "bn1", dst_prefix + "stem/conv1")
    for grpid in range(4):
        num_blocks_per_stage = {
            50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]
        }[cfg.MODEL.RESNETS.DEPTH]
        for blkid in range(num_blocks_per_stage[grpid]):
            _convert_conv(backbone_prefix + f"layer{grpid + 1}.{blkid}.conv1",
                          dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/conv1")
            _convert_norm(backbone_prefix + f"layer{grpid + 1}.{blkid}.bn1",
                          dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/conv1")
            _convert_conv(backbone_prefix + f"layer{grpid + 1}.{blkid}.conv2",
                          dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/conv2")
            _convert_norm(backbone_prefix + f"layer{grpid + 1}.{blkid}.bn2",
                          dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/conv2")
            _convert_conv(backbone_prefix + f"layer{grpid + 1}.{blkid}.conv3",
                          dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/conv3")
            _convert_norm(backbone_prefix + f"layer{grpid + 1}.{blkid}.bn3",
                          dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/conv3")
            if blkid == 0:
                _convert_conv(backbone_prefix + f"layer{grpid + 1}.{blkid}.downsample.0",
                              dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/shortcut")
                _convert_norm(backbone_prefix + f"layer{grpid + 1}.{blkid}.downsample.1",
                              dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/shortcut")
    if has_fpn:
        for lvl in range(2, 6):
            _convert_conv(f"neck.lateral_convs.{lvl-2}.conv", f"backbone/fpn_lateral{lvl}")
            _convert_conv(f"neck.fpn_convs.{lvl-2}.conv", f"backbone/fpn_output{lvl}")

    for i in range(cfg.MODEL.SOLO.MASK_KERNEL_NUM_CONVS):
        _convert_conv(f"bbox_head.cate_convs.{i}.conv", f"mask_kernel/cate_subnet{2*i}")
        _convert_norm(f"bbox_head.cate_convs.{i}.gn", f"mask_kernel/cate_subnet{2*i}")
        _convert_conv(f"bbox_head.kernel_convs.{i}.conv", f"mask_kernel/kernel_subnet{2*i}")
        _convert_norm(f"bbox_head.kernel_convs.{i}.gn", f"mask_kernel/kernel_subnet{2*i}")
        
    _convert_conv("bbox_head.solo_cate", "mask_kernel/solo_cate")
    _convert_conv("bbox_head.solo_kernel", "mask_kernel/solo_kernel")

    for i, in_feature in enumerate(cfg.MODEL.SOLO.MASK_FEATURE_IN_FEATURES):
        head_length = max(1, int(i + 2 - np.log2(cfg.MODEL.SOLO.MASK_FEATURE_COMMON_STRIDE)))
        for k in range(head_length):
            _convert_conv(f"mask_feat_head.convs_all_levels.{i}.conv{k}.conv",
                            f"mask_feature/{in_feature}_{2 * k}")
            _convert_norm(f"mask_feat_head.convs_all_levels.{i}.conv{k}.gn",
                            f"mask_feature/{in_feature}_{2 * k}")
    _convert_conv(f"mask_feat_head.conv_pred.0.conv", f"mask_feature/predictor")
    _convert_norm(f"mask_feat_head.conv_pred.0.gn", f"mask_feature/predictor")

    for k in list(d.keys()):
        if "cell_anchors" in k:
            d.pop(k)
    assert len(d) == 0, d.keys()
    return ret

