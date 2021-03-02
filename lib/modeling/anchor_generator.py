import copy
import math
from typing import List
import tensorflow as tf

from ..layers import Layer, ShapeSpec
from ..structures import box_list
from ..utils.registry import Registry
from ..utils import shape_utils

ANCHOR_GENERATOR_REGISTRY = Registry("ANCHOR_GENERATOR")
"""
Registry for modules that creates object detection anchors for feature maps.
"""


class lazyproperty(object):

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


def _create_grid_offsets(size, stride):
    grid_height, grid_width = size[0], size[1]
    shifts_y = tf.range(0, grid_height * stride, delta=stride)
    shifts_y = tf.cast(shifts_y, dtype=tf.float32)
    shifts_x = tf.range(0, grid_width * stride, delta=stride)
    shifts_x = tf.cast(shifts_x, dtype=tf.float32)
    shift_x, shift_y = tf.meshgrid(shifts_x, shifts_y)
    shift_y = tf.reshape(shift_y, [-1])
    shift_x = tf.reshape(shift_x, [-1])
    return shift_y, shift_x


@ANCHOR_GENERATOR_REGISTRY.register()
class DefaultAnchorGenerator(Layer):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        self.sizes = cfg.MODEL.ANCHOR_GENERATOR.SIZES
        self.aspect_ratios = cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS
        self.strides = [x.stride for x in input_shape]
        # If one size (or aspect ratio) is specified and there are multiple
        # feature maps, then we "broadcast" anchors of that single size
        # (or aspect ratio) over all feature maps.

        self.num_features = len(self.strides)
        if len(self.sizes) == 1:
            self.sizes *= self.num_features
        if len(self.aspect_ratios) == 1:
            self.aspect_ratios *= self.num_features
        assert self.num_features == len(self.sizes)
        assert self.num_features == len(self.aspect_ratios)

    @lazyproperty
    def cell_anchors(self):
        return [self.generate_cell_anchors(s, a) for s, a in zip(self.sizes, self.aspect_ratios)]

    @property
    def box_dim(self):
        """
        Returns:
            int: the dimension of each anchor box.
        """
        return 4

    @lazyproperty
    def num_cell_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
        """
        return [
            shape_utils.combined_static_and_dynamic_shape(cell_anchors)[0]
            for cell_anchors in self.cell_anchors
        ]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(
                grid_sizes, self.strides, self.cell_anchors):
            shift_y, shift_x = _create_grid_offsets(size, stride)
            shifts = tf.stack([shift_y, shift_x, shift_y, shift_x], axis=1)

            anchors.append(
                tf.reshape(
                    tf.add(
                        tf.transpose(tf.reshape(shifts, [1, -1, 4]), [1, 0, 2]),
                        tf.reshape(base_anchors, [1, -1, 4])
                    ),
                    [-1, 4]
                )
            )

        return anchors

    def generate_cell_anchors(self,
                              sizes=(32, 64, 128, 256, 512),
                              aspect_ratios=(0.5, 1, 2)):
        """
        Generate a tensor storing anchor boxes, which are continuous geometric
        rectangles centered on one feature map point sample. We can later build
        the set of anchors for the entire feature map by tiling these tensors;
        see `meth:grid_anchors`.
        Args:
            sizes (tuple[float]): Absolute size of the anchors in the units of
                the input image (the input received by the network, after
                undergoing necessary scaling). The absolute size is given as
                the side length of a box.
            aspect_ratios (tuple[float]]): Aspect ratios of the boxes computed
                as box height / width.
        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing
                anchor boxes in XYXY format.
        """

        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                y0, x0, y1, x1 = -h / 2.0, -w / 2.0, h / 2.0, w / 2.0
                anchors.append([y0, x0, y1, x1])
        return tf.convert_to_tensor(anchors, dtype=tf.float32)

    def call(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to
            generate anchors.
        Returns:
            list[Boxes]: a list of #image elements. Each is a list of
                #feature level Boxes. The Boxes contains anchors of this image
                on the specific feature level.
        """
        grid_sizes = [tf.shape(feature_map)[1:3] for feature_map in features]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)

        anchors = []
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            anchors.append(box_list.BoxList(anchors_per_feature_map))
        return anchors


@ANCHOR_GENERATOR_REGISTRY.register()
class YOLOAnchorGenerator(Layer):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        self.sizes = cfg.MODEL.ANCHOR_GENERATOR.SIZES
        self.strides = [x.stride for x in input_shape]
        self.num_features = len(self.strides)
        assert self.num_features == len(self.sizes)

    @lazyproperty
    def cell_anchors(self):
        return [self.generate_cell_anchors(sizes) for sizes in self.sizes]

    @property
    def box_dim(self):
        """
        Returns:
            int: the dimension of each anchor box.
        """
        return 4

    @lazyproperty
    def num_cell_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
        """
        return [
            shape_utils.combined_static_and_dynamic_shape(cell_anchors)[0]
            for cell_anchors in self.cell_anchors
        ]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(
                grid_sizes, self.strides, self.cell_anchors):
            shift_y, shift_x = _create_grid_offsets(size, stride=1)
            zeros = tf.zeros_like(shift_y)
            shifts = tf.stack([shift_y, shift_x, zeros, zeros], axis=1)
            base_anchors = tf.truediv(base_anchors, float(stride))

            anchors.append(
                tf.reshape(
                    tf.add(
                        tf.transpose(tf.reshape(shifts, [1, -1, 4]), [1, 0, 2]),
                        tf.reshape(base_anchors, [1, -1, 4])
                    ),
                    [-1, 4]
                )
            )

        return anchors

    def generate_cell_anchors(self, sizes):
        """
        Generate a tensor storing anchor boxes, which are continuous geometric
        rectangles centered on one feature map point sample. We can later build
        the set of anchors for the entire feature map by tiling these tensors;
        see `meth:grid_anchors`.
        Args:
            sizes (tuple[float]): Absolute size of the anchors in the units of
                the input image (the input received by the network, after
                undergoing necessary scaling). The absolute size is given as
                the side length of a box.
        Returns:
            Tensor of shape (len(sizes), 4) storing anchor boxes in YXYX format.
        """

        anchors = []
        for width, height in sizes:
            anchors.append([0., 0., height, width])
        return tf.convert_to_tensor(anchors, dtype=tf.float32)

    def call(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to
            generate anchors.
        Returns:
            list[Boxes]: a list of #image elements. Each is a list of
                #feature level Boxes. The Boxes contains anchors of this image
                on the specific feature level.
        """
        grid_sizes = [tf.shape(feature_map)[1:3] for feature_map in features]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)

        anchors = []
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            anchors.append(box_list.BoxList(anchors_per_feature_map))
        return anchors


def build_anchor_generator(cfg, input_shape):
    """
    Built an anchor generator from `cfg.MODEL.ANCHOR_GENERATOR.NAME`.
    """
    anchor_generator = cfg.MODEL.ANCHOR_GENERATOR.NAME
    return ANCHOR_GENERATOR_REGISTRY.get(anchor_generator)(cfg, input_shape)
