from enum import Enum, unique
import tensorflow as tf

from ..utils import shape_utils


class BoxList(object):
    """Box list collection."""

    def __init__(self, boxes):
        """Constructs box collection.

        Args:
            boxes: a tensor of shape [..., 4] representing box corners

        Raises:
            ValueError: if invalid dimensions for bbox data or if bbox data is not in
                float32 format.
        """
        self.data = {}
        self.trackings = {}
        self.boxes = boxes

    @property
    def boxes(self):
        """Convenience function for accessing box coordinates.

        Returns:
            a tensor with shape [N, 4] representing box coordinates.
        """
        return self.get_field('boxes')

    @boxes.setter
    def boxes(self, boxes):
        """Convenience function for setting box coordinates.

        Args:
            boxes: a tensor of shape [N, 4] representing box corners

        Raises:
            ValueError: if invalid dimensions for bbox data
        """
        assert self.data == {}, "BoxList is immutable."
        if boxes.get_shape()[-1] != 4:
            raise ValueError('Invalid dimensions for box data.')
        if boxes.dtype != tf.float32:
            raise ValueError('Invalid tensor type: should be tf.float32')
        self.data['boxes'] = boxes

    def get_all_trackings(self):
        return self.trackings.keys()

    def set_tracking(self, name, value):
        self.trackings[name] = value

    def get_tracking(self, name):
        return self.trackings[name]

    def has_tracking(self, name):
        return name in self.trackings

    def get_all_fields(self):
        """Returns all fields."""
        return self.data.keys()

    def get_extra_fields(self):
        """Returns all non-box fields (i.e., everything not named 'boxes')."""
        return [k for k in self.data.keys() if k != 'boxes']

    def add_field(self, field, field_data):
        """Add field to box list.

        This method can be used to add related box data such as
        weights/labels, etc.

        Args:
            field: a string key to access the data via `get`
            field_data: a tensor containing the data to store in the Boxes
        """
        field_data.get_shape()[:1].assert_is_compatible_with(self.boxes.get_shape()[:1])
        self.data[field] = field_data

    def has_field(self, field):
        return field in self.data

    def get_field(self, field):
        """Accesses a box collection and associated fields.

        This function returns specified field with object; if no field is specified,
        it returns the box coordinates.

        Args:
            field: this optional string parameter can be used to specify
                a related field to be accessed.

        Returns:
            a tensor representing the box collection or an associated field.

        Raises:
            ValueError: if invalid field
        """
        if not self.has_field(field):
            raise ValueError('field ' + str(field) + ' does not exist')
        return self.data[field]

    def set_field(self, field, field_data):
        """Sets the value of a field.

        Updates the field of a Boxes with a given value.

        Args:
            field: (string) name of the field to set value.
            value: the value to assign to the field.

        Raises:
            ValueError: if the Boxes does not have specified field.
        """
        if not self.has_field(field):
            raise ValueError('field %s does not exist' % field)
        field_data.get_shape()[:1].assert_is_compatible_with(self.boxes.get_shape()[:1])
        self.data[field] = field_data

    def as_tensor_dict(self, fields=None, trackings=None):
        """Retrieves specified fields as a dictionary of tensors.

        Args:
        fields: (optional) list of fields to return in the dictionary.
            If None (default), all fields are returned.

        Returns:
            tensor_dict: A dictionary of tensors specified by fields.

        Raises:
            ValueError: if specified field is not contained in Boxes.
        """
        tensor_dict = {}
        if fields is None:
            fields = self.get_all_fields()
        if trackings is None:
            trackings = self.get_all_trackings()
        for field in fields:
            if not self.has_field(field):
                raise ValueError('Boxes must contain all specified fields')
            tensor_dict[field] = self.get_field(field)
        for tracking in trackings:
            if not self.has_tracking(tracking):
                raise ValueError('Boxes must contain all specified trackings')
            tensor_dict[tracking] = self.get_tracking(tracking)
        return tensor_dict

    @classmethod
    def from_tensor_dict(cls, tensor_dict, fields=None, trackings=None):
        """Construct a new BoxList object from tensor_dict.

        Args:
            tensor_dict: a dict hold data to construct a new object.

        Raises:
            ValueError: if boxes or num_boxes not in the tensor_dict.
        """
        boxlist = cls(tensor_dict.pop("boxes"))
        if trackings is not None:
            for tracking in trackings:
                boxlist.set_tracking(tracking, tensor_dict.pop(tracking))
        if fields is not None:
            for field in fields:
                boxlist.add_field(field, tensor_dict[field])
        else:
            for field in tensor_dict:
                boxlist.add_field(field, tensor_dict[field])
        return boxlist


class SparseBoxList(object):

    def __init__(self, indices, data, dense_shape):
        assert isinstance(data, BoxList), type(data)
        self.data = data
        self.dense_shape = dense_shape
        self.indices = indices
        self.trackings = {}

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, indices):
        indices.get_shape()[0].assert_is_compatible_with(self.data.boxes.get_shape()[0])
        self._indices = indices

    def get_all_trackings(self):
        return self.trackings.keys()

    def set_tracking(self, name, value):
        self.trackings[name] = value

    def get_tracking(self, name):
        return self.trackings[name]

    def has_tracking(self, name):
        return name in self.trackings

    def to_dense(self, scope=None):
        with tf.name_scope(scope, 'BoxListSparseToDense'):
            tensor_dict = {}
            flatten_indices = self.indices[:, 0] * self.dense_shape[1] + self.indices[:, 1]
            for field in self.data.get_all_fields():
                if field != "is_valid":
                    sparse_field_data = self.data.get_field(field)
                    field_specific_shape = shape_utils.combined_static_and_dynamic_shape(
                        sparse_field_data)[1:]

                    if len(field_specific_shape) > 0:
                        flatten_dense_field_data = tf.IndexedSlices(
                            values=sparse_field_data,
                            indices=flatten_indices,
                            dense_shape=tf.concat(
                                [
                                    [self.dense_shape[0] * self.dense_shape[1]],
                                    tf.cast(field_specific_shape, tf.int64)
                                ],
                                axis=0
                            )
                        )
                        tensor_dict[field] = tf.reshape(
                            tf.convert_to_tensor(flatten_dense_field_data),
                            [self.dense_shape[0], self.dense_shape[1]] + field_specific_shape
                        )
                    else:
                        tensor_dict[field] = tf.sparse.to_dense(
                            tf.SparseTensor(
                                self.indices, sparse_field_data, self.dense_shape
                            )
                        )
            mask = tf.SparseTensor(
                self.indices,
                tf.ones([tf.shape(self.indices)[0]], dtype=tf.bool),
                self.dense_shape
            )
            mask = tf.sparse.to_dense(mask, default_value=False)
            tensor_dict['is_valid'] = mask
            dense = BoxList.from_tensor_dict(tensor_dict)
            for tracking in self.get_all_trackings():
                dense.set_tracking(tracking, self.get_tracking(tracking))
            return dense

    @classmethod
    def from_dense(cls, boxlist, scope=None):
        with tf.name_scope(scope, 'BoxListSparseFromDense'):
            assert boxlist.has_field('is_valid')
            tensor_dict = {}
            dense_shape = tf.cast(tf.shape(boxlist.boxes)[:-1], tf.int64)
            indices = tf.where(boxlist.get_field('is_valid'))
            for field in boxlist.get_all_fields():
                if field != 'is_valid':
                    tensor_dict[field] = tf.gather_nd(
                        boxlist.get_field(field), indices
                    )
            data = BoxList.from_tensor_dict(tensor_dict)
            sparse = cls(indices, data, dense_shape)
            for tracking in boxlist.get_all_trackings():
                sparse.set_tracking(tracking, boxlist.get_tracking(tracking))
            return sparse
