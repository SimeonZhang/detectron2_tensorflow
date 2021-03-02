from __future__ import division
import tensorflow as tf

from ..utils import shape_utils


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    Attributes:
        image_shapes [N, 2]
    """

    def __init__(
        self,
        tensor,
        image_shapes,
        size_divisibility=0,
        padding_value=0
    ):
        """
        Arguments:
            tensor: of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_shapes [N, 2].
        """
        self.tensor = tensor
        if image_shapes.get_shape()[0].value is not None:
            self._num_images = image_shapes.get_shape()[0].value
        else:
            self._num_images = tf.shape(image_shapes)[0]
        self.image_shapes = image_shapes
        self.size_divisibility = size_divisibility
        self.padding_value = padding_value

    @property
    def num_images(self):
        return self._num_images

    def __getitem__(self, idx):
        """
        Access the individual image in its original shape.
        Returns:
            Tensor: an image of shape (H, W) or (H, W, C)
        """
        original_shape = self.image_shapes[idx]
        return self.tensor[idx, :original_shape[0], :original_shape[1], ...]

    @classmethod
    def from_tensors(
        cls,
        tensors,
        image_shapes,
        size_divisibility=0,
        pad_value=0.
    ):
        """
        Args:
            tensor: a batch of image from `tf.data.Dataset.padded_batch` ,
                if size_divisibility > 0, will be padded to match the requirements.
            size_divisibility (int): If `size_divisibility > 0`, also adds padding to ensure
                the common height and width is divisible by `size_divisibility`
            pad_value (float): value to pad
        Returns:
            an `ImageList`.
        """
        if pad_value != 0:

            def replace_paddings_per_image(args):
                tensor, image_shape = args
                orig_shape = tf.shape(tensor)[:2]
                pad_shape = orig_shape - image_shape
                if tensor.shape.ndims == 2:
                    paddings = [[0, pad_shape[0]], [0, pad_shape[1]]]
                else:
                    paddings = [[0, pad_shape[0]], [0, pad_shape[1]], [0, 0]]
                tensor = tensor[:image_shape[0], :image_shape[1]]
                tensor = tf.pad(tensor, paddings, constant_values=pad_value)
                return tensor

            tensors = tf.map_fn(
                replace_paddings_per_image,
                [tensors, tf.convert_to_tensor(image_shapes)],
                dtype=tensors.dtype
            )

        if size_divisibility > 0:
            stride = size_divisibility
            orig_shape = tf.shape(tensors)[1:3]
            max_size = tf.cast(
                tf.ceil(tf.cast(orig_shape, tf.float32) / stride) * stride, tf.int32)
            pad_shape = max_size - orig_shape
            if tensors.shape.ndims == 3:
                paddings = [[0, 0], [0, pad_shape[0]], [0, pad_shape[1]]]
            else:
                paddings = [[0, 0], [0, pad_shape[0]], [0, pad_shape[1]], [0, 0]]
            tensors = tf.pad(tensors, paddings, constant_values=pad_value)

        return ImageList(tensors, image_shapes)
