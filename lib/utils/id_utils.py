import tensorflow as tf
import numpy as np


def rgb2id(color):
    if isinstance(color, tf.Tensor) and len(color.shape) == 3:
        if color.dtype == tf.uint8:
            color = tf.cast(color, tf.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, tf.Tensor):
        rgb_map = []
        for i in range(3):
            rgb_map.append(id_map % 256)
            id_map //= 256
        rgb_map = tf.stack(rgb_map, axis=-1)
        return rgb_map
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color
