import math
import sys
import tensorflow as tf

from ..layers import Layer, ROIAlign
from ..structures import box_list_ops

__all__ = ["ROIPooler"]


def assign_boxes_to_levels(boxlist,
                           min_level,
                           max_level,
                           canonical_box_size,
                           canonical_level):
    """
    Map each box in `box_lists` to a feature map level index and return
    the assignment vector.
    Args:
        boxlist (BoxList): A list of N Boxes or
            N RotatedBoxes, where N is the number of images in the batch.
        min_level (int): Smallest feature map level index. The input is
            considered index 0, the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.
        canonical_box_size (int): A canonical box size in pixels
            (sqrt(box area)).
        canonical_level (int): The feature map level index on which a
            canonically-sized box should be placed.
    Returns:
        A tensor of length M, where M is the total number of boxes aggregated
            over all N batch images. The memory layout corresponds to the
            concatenation of boxes from all images.
            Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the
            box is at `self.min_level + i`).
    """
    eps = sys.float_info.epsilon
    box_sizes = tf.sqrt(box_list_ops.area(boxlist))
    # Eqn.(1) in FPN paper
    level_assignments = tf.cast(
        tf.floor(
            canonical_level + tf.log(box_sizes / canonical_box_size + eps) / math.log(2)
        ),
        tf.int64
    )
    level_assignments = tf.clip_by_value(
        level_assignments, clip_value_min=min_level, clip_value_max=max_level
    )
    return level_assignments - min_level


class ROIPooler(Layer):
    """
    Region of interest feature map pooler that supports pooling from one or
    more feature maps.
    """

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the
                pooled region, e.g., 14 x 14. If tuple or list is given,
                the length must be 2.
            scales (list[float]): The scale for each low-level pooling op
                relative to the input image. For a feature map with stride s
                relative to the input image, scale is defined as a 1 / s.
            sampling_ratio (int): The `sampling_ratio` parameter for the
                ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that
                should be applied. For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels
                (sqrt(box area)). The default is heuristically defined as 224
                pixels in the FPN paper (based on ImageNet pre-training).
            canonical_level (int): The feature map level index on which a
                canonically-sized box should be placed. The default is defined
                as level 4 in the FPN paper.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        if pooler_type == "ROIAlign":
            self.level_poolers = [
                ROIAlign(
                    output_size, spatial_scale=scale,
                    sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            ]
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = [
                ROIAlign(
                    output_size, spatial_scale=scale,
                    sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            ]
        # elif pooler_type == "ROIAlignRotated":
        #     self.level_poolers = [
        #         ROIAlignRotated(
        #             output_size, spatial_scale=scale,
        #             sampling_ratio=sampling_ratio
        #         )
        #         for scale in scales
        #     ]
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])
        assert math.isclose(min_level, int(min_level)) and math.isclose(max_level, int(max_level))
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert 0 < self.min_level and self.min_level <= self.max_level
        assert self.min_level <= canonical_level and canonical_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    def call(self, x, instances):
        """
        Args:
            x (list[Tensor]): A list of feature maps with scales matching
                those used to construct this module.
            instances (SparseBoxList):
        Returns:
            Tensor:
                A tensor of shape (M, output_size, output_size, C) where M is
                the total number of boxes aggregated over all N batch images
                and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        assert len(x) == num_level_assignments, (
            "unequal value, num_level_assignments={}, but x is list of {} "
            "Tensors".format(num_level_assignments, len(x)))

        if num_level_assignments == 1:
            return self.level_poolers[0](
                x[0], instances.data.boxes, instances.indices[:, 0]
            )

        level_assignments = assign_boxes_to_levels(
            instances.data,
            self.min_level,
            self.max_level,
            self.canonical_box_size,
            self.canonical_level
        )

        output_inds, output_data = [], []
        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            level_inds = tf.where(tf.equal(level_assignments, level))[:, 0]
            level_boxes = tf.gather(instances.data.boxes, level_inds)
            level_box_inds = tf.gather(instances.indices[:, 0], level_inds)
            level_data = pooler(x_level, level_boxes, level_box_inds)
            output_inds.append(tf.cast(level_inds, tf.int32))
            output_data.append(level_data)
            tf.summary.scalar(
                'roi_align/num_roi_level_{}'.format(level + 2), tf.size(level_inds))
        output_data = tf.concat(output_data, axis=0)
        output_inds = tf.concat(output_inds, axis=0)
        output_inds_invert_perm = tf.invert_permutation(output_inds)
        output = tf.gather(output_data, output_inds_invert_perm)

        return output
