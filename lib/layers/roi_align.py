import tensorflow as tf

from .base import Layer
from .functional import crop_and_resize

slim = tf.contrib.slim


class ROIAlign(Layer):
    def __init__(self,
                 output_size,
                 spatial_scale,
                 sampling_ratio,
                 aligned=True):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each
                output sample.
            aligned (bool): if False, use the original tensorflow op
                `crop_and_resize`. If True, align the results more perfectly.

        Note:
            The meaning of aligned=True:

            The way tf.image.crop_and_resize works (with normalized box):
            Initial point (the value of output[0]): x0_box * (W_img - 1)
            Spacing: w_box * (W_img - 1) / (W_crop - 1)
            Use the above grid to bilinear sample.

            With `aligned=True`:
            Spacing: w_box / W_crop
            Initial point: x0_box + spacing/2 - 0.5
            (-0.5 because bilinear sample (in my definition) assumes floating
                point coordinate (0.0, 0.0) is the same as pixel value (0, 0))
        """
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        assert isinstance(sampling_ratio, int), sampling_ratio
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def call(self, inputs, boxes, box_inds):
        """
        Args:
            inputs: NHWC images
            boxes: Bx4 boxes.

        """
        crop_size = self.output_size
        if self.sampling_ratio > 0:
            crop_size = [x * self.sampling_ratio for x in crop_size]
        scaled_boxes = boxes * self.spatial_scale

        ret = crop_and_resize(
            inputs, scaled_boxes, box_inds, crop_size, self.aligned)
        if self.sampling_ratio > 0:
            ret = slim.avg_pool2d(
                ret,
                kernel_size=self.sampling_ratio,
                stride=self.sampling_ratio,
                padding="SAME"
            )
        return ret

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr
