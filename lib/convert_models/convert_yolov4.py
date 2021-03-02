import os
import json
import numpy as np


def get_weight_map(cfg):
    weight_path = os.path.join(cfg.PRETRAINS.ROOT, cfg.PRETRAINS.DARKNET)
    node_path = weight_path.split(".")
    node_path[-1] = "json"
    node_path = ".".join(node_path)

    with open(node_path) as node_fp:
        node_cfg = json.load(node_fp)

    weight_map = {}
    with open(weight_path, "rb") as weight_fp:
        data = np.fromfile(weight_fp, dtype=np.float32)
        start = 0
        for node in node_cfg["nodes"]:
            name = node["name"]
            in_channels = node["in_channels"]
            out_channels = node["out_channels"]
            kernel_size = node["size"]

            bias = data[start: start + out_channels]
            start += out_channels

            norm = node_cfg["norm"].get(name)
            if norm:
                gamma = data[start: start + out_channels]
                start += out_channels
                weight_map[name + "/norm/beta"] = bias
                weight_map[name + "/norm/gamma"] = gamma
                if norm == "bn":
                    mean = data[start: start + out_channels]
                    start += out_channels
                    var = data[start: start + out_channels]
                    start += out_channels
                    weight_map[name + "/norm/moving_mean"] = mean
                    weight_map[name + "/norm/moving_variance"] = var
            else:
                weight_map[name + "/bias"] = bias

            weight_size = in_channels * out_channels * kernel_size * kernel_size
            weight = data[start: start + weight_size]
            start += weight_size
            weight = np.reshape(
                weight, [out_channels, in_channels, kernel_size, kernel_size]
            )
            weight = np.transpose(weight, [2, 3, 1, 0])
            weight_map[name + "/weights"] = weight
        assert data.shape[0] == start, (data.shape[0] - start)

        def get_box_indices(num_anchors, num_classes):
            idxs = np.arange(num_anchors * (5 + num_classes))
            idx_xmin = np.arange(num_anchors) * (5 + num_classes)
            idx_ymin = idx_xmin + 1
            idx_xmax = idx_xmin + 2
            idx_ymax = idx_xmin + 3
            idxs[idx_xmin] = idx_ymin
            idxs[idx_ymin] = idx_xmin
            idxs[idx_xmax] = idx_ymax
            idxs[idx_ymax] = idx_xmax
            return idxs

        idxs = get_box_indices(3, 80)
        for i in range(3):
            name = f"head/pred{i+1}"
            v = weight_map[name + "/bias"]
            weight_map[name + "/bias"] = v[idxs]
            v = weight_map[name + "/weights"]
            weight_map[name + "/weights"] = v[..., idxs]

    return weight_map
