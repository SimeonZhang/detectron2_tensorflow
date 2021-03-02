import numpy as np


def convert_weights(d, cfg):
    has_fpn = cfg.MODEL.NECK.NAME == "FPN"
    use_res5_in_stage2 = cfg.MODEL.ROI_HEADS.NAME == "Res5ROIHeads"
    is_retina = cfg.MODEL.NECK.TOP_BLOCK_TYPE == "P6P7"
    ret = {}

    def _convert_conv(src, dst):
        src_w = d.pop(src + ".weight").transpose(2, 3, 1, 0)
        ret[dst + "/weights"] = src_w
        if src + ".norm.weight" in d:     # has norm
            ret[dst + "/norm/gamma"] = d.pop(src + ".norm.weight")
            ret[dst + "/norm/beta"] = d.pop(src + ".norm.bias")
        if src + ".norm.running_var" in d:    # batch norm
            ret[dst + "/norm/moving_variance"] = d.pop(src + ".norm.running_var")
            ret[dst + "/norm/moving_mean"] = d.pop(src + ".norm.running_mean")
            if src + ".norm.num_batches_tracked" in d:
                d.pop(src + ".norm.num_batches_tracked")
        if src + "_offset.weight" in d:
            ret[dst + "/offset_weights"] = d.pop(src + "_offset.weight").transpose(2, 3, 1, 0)
            ret[dst + "/offset_bias"] = d.pop(src + "_offset.bias")
        if src + ".bias" in d:
            ret[dst + "/bias"] = d.pop(src + ".bias")

    def _convert_fc(src, dst):
        ret[dst + "/weights"] = d.pop(src + ".weight").transpose()
        ret[dst + "/bias"] = d.pop(src + ".bias")

    if has_fpn:
        backbone_prefix = "backbone.bottom_up."
        dst_prefix = "backbone/"
    else:
        backbone_prefix = "backbone."
        dst_prefix = "backbone/"
    _convert_conv(backbone_prefix + "stem.conv1", dst_prefix + "stem/conv1")
    for grpid in range(4):
        if use_res5_in_stage2 and grpid == 3 and not is_retina:
            backbone_prefix = "roi_heads."
            dst_prefix = "roi_heads/"
        num_blocks_per_stage = {
            50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]
        }[cfg.MODEL.RESNETS.DEPTH]
        for blkid in range(num_blocks_per_stage[grpid]):
            _convert_conv(backbone_prefix + f"res{grpid + 2}.{blkid}.conv1",
                          dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/conv1")
            _convert_conv(backbone_prefix + f"res{grpid + 2}.{blkid}.conv2",
                          dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/conv2")
            _convert_conv(backbone_prefix + f"res{grpid + 2}.{blkid}.conv3",
                          dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/conv3")
            if blkid == 0:
                _convert_conv(backbone_prefix + f"res{grpid + 2}.{blkid}.shortcut",
                              dst_prefix + f"res{grpid + 2}/block_{blkid + 1}/shortcut")
    if is_retina:
        for lvl in range(6, 8):
            _convert_conv(f"backbone.top_block.p{lvl}", f"neck/top_block/p{lvl}")
        for lvl in range(3, 6):
            _convert_conv(f"backbone.fpn_lateral{lvl}", f"neck/fpn_lateral{lvl}")
            _convert_conv(f"backbone.fpn_output{lvl}", f"neck/fpn_output{lvl}")
    elif has_fpn:
        for lvl in range(2, 6):
            _convert_conv(f"backbone.fpn_lateral{lvl}", f"neck/fpn_lateral{lvl}")
            _convert_conv(f"backbone.fpn_output{lvl}", f"neck/fpn_output{lvl}")

    def get_box_indices(num_reg_classes):
        idx_xmin = np.arange(num_reg_classes) * 4
        idx_ymin = idx_xmin + 1
        idx_xmax = idx_xmin + 2
        idx_ymax = idx_xmin + 3
        idxs = np.stack([idx_ymin, idx_xmin, idx_ymax, idx_xmax], axis=-1)
        idxs = np.reshape(idxs, [num_reg_classes * 4])
        return idxs

    if is_retina:
        for i in range(cfg.MODEL.RETINANET.NUM_CONVS):
            _convert_conv(f"head.cls_subnet.{2*i}", f"head/cls_subnet{2*i}")
            _convert_conv(f"head.bbox_subnet.{2*i}", f"head/bbox_subnet{2*i}")

        _convert_conv("head.cls_score", "head/cls_score")
        _convert_conv("head.bbox_pred", "head/bbox_pred")
        num_anchors = ret["head/bbox_pred/bias"].shape[0] // 4
        idxs = get_box_indices(num_anchors)
        v = ret["head/bbox_pred/bias"]
        ret["head/bbox_pred/bias"] = v[idxs]
        v = ret["head/bbox_pred/weights"]
        ret["head/bbox_pred/weights"] = v[..., idxs]

    elif cfg.MODEL.META_ARCHITECTURE != "SemanticSegmentor":
        # RPN:
        def _convert_rpn(src, dst):
            _convert_conv(src + ".conv", dst + "/share")
            _convert_conv(src + ".objectness_logits", dst + "/objectness_logits")
            _convert_conv(src + ".anchor_deltas", dst + "/anchor_deltas")
            num_anchors = ret[dst + "/objectness_logits/bias"].shape[0]
            idxs = get_box_indices(num_anchors)
            v = ret[dst + "/anchor_deltas/bias"]
            ret[dst + "/anchor_deltas/bias"] = v[idxs]
            v = ret[dst + "/anchor_deltas/weights"]
            ret[dst + "/anchor_deltas/weights"] = v[..., idxs]

        _convert_rpn("proposal_generator.rpn_head", "proposal_generator/rpn_head")

        def _convert_box_predictor(src, dst):
            num_reg_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            if cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG:
                num_reg_classes = 1
            idxs = get_box_indices(num_reg_classes)
            v = d.pop(src + ".bbox_pred.bias")
            ret[dst + "/box_deltas/bias"] = v[idxs]
            v = d.pop(src + ".bbox_pred.weight")
            ret[dst + "/box_deltas/weights"] = v.transpose()[..., idxs]

            _convert_fc(src + ".cls_score", dst + "/class_logits")

        # Fast R-CNN: box head
        has_cascade = cfg.MODEL.ROI_HEADS.NAME in ["CascadeROIHeads", "CascadeLCCHeads"]
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        fc_in_channels = cfg.MODEL.NECK.OUT_CHANNELS
        if not has_fpn:
            fc_in_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * 2 ** 3
        if cfg.MODEL.ROI_BOX_HEAD.NUM_CONV > 0:
            fc_in_channels = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        fc_out_channels = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        if has_cascade:
            assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
            for k in range(3):
                for i in range(cfg.MODEL.ROI_BOX_HEAD.NUM_CONV):
                    _convert_conv(f"roi_heads.box_head.{k}.conv{i+1}",
                                  f"roi_heads/box_head_stage{k+1}/conv{i+1}")
                for i in range(cfg.MODEL.ROI_BOX_HEAD.NUM_FC):
                    _convert_fc(f"roi_heads.box_head.{k}.fc{i+1}",
                                f"roi_heads/box_head_stage{k+1}/fc{i+1}")
                    if i == 0:
                        w = ret[f"roi_heads/box_head_stage{k+1}/fc{i+1}/weights"]
                        w = w.reshape(
                            [fc_in_channels, pooler_resolution, pooler_resolution, fc_out_channels]
                        )
                        w = w.transpose(1, 2, 0, 3)
                        w = w.reshape(
                            [
                                pooler_resolution * pooler_resolution * fc_in_channels,
                                fc_out_channels
                            ]
                        )
                        ret[f"roi_heads/box_head_stage{k+1}/fc{i+1}/weights"] = w
                _convert_box_predictor(f"roi_heads.box_predictor.{k}", f"roi_heads/box_predictor_stage{k+1}")
        else:
            for i in range(cfg.MODEL.ROI_BOX_HEAD.NUM_CONV):
                _convert_conv(f"roi_heads.box_head.conv{i+1}", f"roi_heads/box_head/conv{i+1}")
            for i in range(cfg.MODEL.ROI_BOX_HEAD.NUM_FC):
                _convert_fc(f"roi_heads.box_head.fc{i+1}", f"roi_heads/box_head/fc{i+1}")
                if i == 0:
                    w = ret[f"roi_heads/box_head/fc{i+1}/weights"]
                    w = w.reshape(
                        [fc_in_channels, pooler_resolution, pooler_resolution, fc_out_channels]
                    )
                    w = w.transpose(1, 2, 0, 3)
                    w = w.reshape(
                        [pooler_resolution * pooler_resolution * fc_in_channels, fc_out_channels]
                    )
                    ret[f"roi_heads/box_head/fc{i+1}/weights"] = w
            dst = "roi_heads/fastrcnn" if use_res5_in_stage2 else "roi_heads/box_predictor"
            _convert_box_predictor("roi_heads.box_predictor", dst)

        # mask head
        if cfg.MODEL.MASK_ON:
            for fcn in range(cfg.MODEL.ROI_MASK_HEAD.NUM_CONV):
                _convert_conv(f"roi_heads.mask_head.mask_fcn{fcn+1}",
                              f"roi_heads/mask_head/mask_fcn{fcn+1}")
            _convert_conv("roi_heads.mask_head.deconv", "roi_heads/mask_head/deconv")
            _convert_conv("roi_heads.mask_head.predictor", "roi_heads/mask_head/predictor")

    # semantic segmentation head
    if cfg.MODEL.META_ARCHITECTURE in ["PanopticFPN", "SemanticSegmentor"]:
        for i, in_feature in enumerate(cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES):
            head_length = max(1, int(i + 2 - np.log2(cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE)))
            for k in range(head_length):
                _convert_conv(f"sem_seg_head.{in_feature}.{2 * k}",
                              f"sem_seg_head/{in_feature}_{2 * k}")
        _convert_conv(f"sem_seg_head.predictor", f"sem_seg_head/predictor")

    for k in list(d.keys()):
        if "cell_anchors" in k:
            d.pop(k)
    assert len(d) == 0, d.keys()
    return ret
