

def convert_weights(d, cfg):
    ret = {}

    def _convert_bn(src, dst):
        if src + "_bn_s" in d:
            ret[dst + "/norm/gamma"] = d.pop(src + "_bn_s")
            ret[dst + "/norm/beta"] = d.pop(src + "_bn_b")
            ret[dst + "/norm/moving_variance"] = d.pop(src + "_bn_riv")
            ret[dst + "/norm/moving_mean"] = d.pop(src + "_bn_rm")
        elif src + "_bn_gamma" in d:
            ret[dst + "/norm/gamma"] = d.pop(src + "_bn_gamma")
            ret[dst + "/norm/beta"] = d.pop(src + "_bn_beta")
            ret[dst + "/norm/moving_variance"] = d.pop(src + "_bn_running_var")
            ret[dst + "/norm/moving_mean"] = d.pop(src + "_bn_running_mean")

    def _convert_conv(src, dst):
        src_w = d.pop(src + "_w").transpose(2, 3, 1, 0)
        ret[dst + "/weights"] = src_w
        _convert_bn(src, dst)
        if src + "_b" in d:
            ret[dst + "/bias"] = d.pop(src + "_b")

    dst_prefix = "backbone/"
    _convert_conv("conv1", dst_prefix + "stem/conv1")
    _convert_bn("res_conv1", dst_prefix + "stem/conv1")
    for grpid in range(4):
        for blkid in range([3, 4, 6 if cfg.MODEL.RESNETS.DEPTH == 50 else 23, 3][grpid]):
            _convert_conv(f"res{grpid + 2}_{blkid}_branch2a",
                          dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/conv1")
            _convert_conv(f"res{grpid + 2}_{blkid}_branch2b",
                          dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/conv2")
            _convert_conv(f"res{grpid + 2}_{blkid}_branch2c",
                          dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/conv3")
            if blkid == 0:
                _convert_conv(f"res{grpid + 2}_{blkid}_branch1",
                              dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/shortcut")

    assert len(d) == 2, d.keys()
    return ret
