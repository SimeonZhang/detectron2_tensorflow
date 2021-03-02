"""Numpy BoxMaskList classes and functions."""

import numpy as np
from . import np_box_list


class BoxMaskList(np_box_list.BoxList):
    """Convenience wrapper for BoxList with masks.

    BoxMaskList extends the np_box_list.BoxList to contain masks as well.
    In particular, its constructor receives both boxes and masks. Note that the
    masks correspond to the full image.
    """

    def __init__(self, box_data, mask_data):
        """Constructs box collection.

        Args:
            box_data: a numpy array of shape [N, 4] representing box coordinates
            mask_data: a numpy array of shape [N, height, width] representing masks
                with values are in {0,1}. The masks correspond to the full
                image. The height and the width will be equal to image height and width.

        Raises:
            ValueError: if bbox data is not a numpy array
            ValueError: if invalid dimensions for bbox data
            ValueError: if mask data is not a numpy array
            ValueError: if invalid dimension for mask data
        """
        super(BoxMaskList, self).__init__(box_data)
        if not isinstance(mask_data, np.ndarray):
            raise ValueError('Mask data must be a numpy array.')
        if len(mask_data.shape) != 3:
            raise ValueError('Invalid dimensions for mask data.')
        if mask_data.dtype != np.uint8:
            raise ValueError('Invalid data type for mask data: uint8 is required.')
        if mask_data.shape[0] != box_data.shape[0]:
            raise ValueError('There should be the same number of boxes and masks.')
        self.data['masks'] = mask_data

    def get_masks(self):
        """Convenience function for accessing masks.

        Returns:
            a numpy array of shape [N, height, width] representing masks
        """
        return self.get_field('masks')
