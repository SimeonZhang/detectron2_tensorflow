import tensorflow as tf

from ...structures import box_list
from ...structures import box_list_ops
from ...layers import Layer, Conv2D, Linear
from ...utils.shape_utils import combined_static_and_dynamic_shape

slim = tf.contrib.slim

__all__ = ["ObjectRelationModule", "compute_geometry_embeddings", "compute_rank_embeddings"]


def compute_rank_embeddings(num_boxes, embedding_dim=128, wave_length=1000):
    assert embedding_dim % 2 == 0, embedding_dim

    rank_range = tf.range(float(num_boxes))
    feature_range = tf.range(float(dimension / 2))
    wave_length = tf.cast(wave_length, tf.float32)
    dim_matrix = tf.pow(wave_length, (2. / embedding_dim) * feature_range)

    dim_matrix = tf.reshape(dim_matrix, [1, -1])
    rank_matrix = tf.expand_dims(rank_range, axis=1)
    div_matrix = tf.div(rank_matrix, dim_matrix)
    embeddings = tf.concat([tf.sin(div_matrix), tf.cos(div_matrix)], axis=1)
    embeddings = tf.expand_dims(embeddings, axis=0)
    return embeddings # [1, R, embedding_dim]


def compute_geometry_embeddings(
    proposals,
    embedding_field="geometry_embeddings",
    embedding_dim=64,
    wave_length=1000
):
    """ 
    Compute geometry embeddings.
    Args:
        proposals: SparseBoxList representing the proposals

    Returns:
        proposals: proposals with field `embedding_field`.
    """
    assert embedding_dim % 8 == 0, embedding_dim
    assert isinstance(proposals, box_list.SparseBoxList), proposals
    with tf.name_scope("ComputeGeometryEmbeddings"):
        dense_proposals = proposals.to_dense()
        # 1. compute 4-d geometry feature of shape [N, R, R, 4]
        ymin, xmin, ymax, xmax = tf.split(dense_proposals.boxes, 4, axis=2) # [N, R, 1]
        height = ymax - ymin + 1.
        width = xmax - xmin + 1.
        center_y = 0.5 * (ymin + ymax)
        center_x = 0.5 * (xmin + xmax)

        delta_y = tf.truediv(
            center_y - tf.transpose(center_y, [0, 2, 1]), height
        ) # [N, R, R]
        delta_y = tf.log(tf.maximum(delta_y, 1e-5))
        delta_x = tf.truediv(
            center_x - tf.transpose(center_x, [0, 2, 1]), width
        )
        delta_x = tf.log(tf.maximum(delta_x, 1e-5))
        delta_height = tf.log(tf.truediv(height, tf.transpose(height, [0, 2, 1])))
        delta_width = tf.log(tf.truediv(width, tf.transpose(width, [0, 2, 1])))

        concat_list = [delta_y, delta_x, delta_height, delta_width]
        for idx, value in enumerate(concat_list):
            concat_list[idx] = tf.expand_dims(value, axis=3) # [N, R, R, 1]
        position_matrix = tf.concat(concat_list, axis=3) # [N, R, R, 4]

        # 2. embed the 4-d feature a high-dimensional representation
        wave_length = tf.cast(wave_length, tf.float32)
        feat_range = tf.range(float(embedding_dim / 8.))
        dim_matrix = tf.pow(wave_length, (8. / embedding_dim) * feat_range)

        dim_matrix = tf.reshape(dim_matrix, [1, 1, 1, 1, -1])
        # position_matrix, [N, R, R, 4, 1]
        position_matrix = tf.expand_dims(100.0 * position_matrix, axis=4)
        # div_matrix, [N, R, R, 4, embedding_dim / 8]
        div_matrix = tf.div(position_matrix, dim_matrix)
        # embeddings, [N, R, R, 4, embedding_dim / 4]
        embeddings = tf.concat([tf.sin(div_matrix), tf.cos(div_matrix)], axis=4)
        embedding_shape = tf.shape(embeddings)
        embeddings = tf.reshape(
            embeddings,
            [embedding_shape[0], embedding_shape[1], embedding_shape[2], embedding_dim]
        )
        # embeddings, [N, R, R, embedding_dim]
        sparse_embeddings = tf.gather_nd(embeddings, proposals.indices)
        proposals.data.add_field(embedding_field, sparse_embeddings)


class ObjectRelationModule(Layer):

    def __init__(
        self,
        input_size,
        embedding_dim=64,
        key_dim=64,
        num_groups=16,
        **kwrags
    ):
        super().__init__(
            input_size=input_size,
            embedding_dim=embedding_dim,
            key_dim=key_dim,
            num_groups=num_groups,
            **kwrags
        )
        self.build()

    def build(self):
        assert self.key_dim % self.num_groups == 0, (
            f"`key_dim`({self.key_dim}) is not divisible by `num_groups`({self.num_groups}).")
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            self.geometry = Conv2D(
                in_channels=self.embedding_dim,
                out_channels=self.num_groups,
                kernel_size=1,
                activation=tf.nn.relu,
                scope="geometry"
            )
            self.query = Linear(
                self.input_size,
                self.key_dim,
                activation=None,
                scope="query"
            )
            self.key = Linear(
                self.input_size,
                self.key_dim,
                activation=None,
                scope="key"
            )
            self.value = Linear(
                self.input_size,
                int(self.input_size / self.num_groups),
                activation=None,
                scope="value"
            )

    def call(self, features, instances):
        if not instances.data.has_field("geometry_embeddings"):
            compute_geometry_embeddings(
                instances, "geometry_embeddings", self.embedding_dim
            )
        query = self.query(features)
        key = self.key(features)
        value = self.value(features)

        def add_or_set_field(instances, field, value):
            if instances.data.has_field(field):
                instances.data.set_field(field, value)
            else:
                instances.data.add_field(field, value)

        add_or_set_field(instances, "query", query)
        add_or_set_field(instances, "key", key)
        add_or_set_field(instances, "value", value)

        dense_instances = instances.to_dense()

        geometry_embeddings = dense_instances.get_field("geometry_embeddings")
        N, R, *_ = combined_static_and_dynamic_shape(geometry_embeddings)
        geometry_weight = self.geometry(geometry_embeddings)
        geometry_weight = tf.transpose(geometry_weight, [0, 1, 3, 2]) # [N, R, num_group, R]

        query = tf.reshape(
            dense_instances.get_field("query"), [N, R, self.num_groups, int(self.key_dim / self.num_groups)]
        )
        query = tf.transpose(query, [0, 2, 1, 3])

        key = tf.reshape(
            dense_instances.get_field("key"), [N, R, self.num_groups, int(self.key_dim / self.num_groups)]
        )
        key = tf.transpose(key, [0, 2, 1, 3])

        dot = tf.matmul(query, key, transpose_a=False, transpose_b=True)
        scaled_dot = dot / tf.sqrt(tf.cast(self.key_dim / self.num_groups, tf.float32))
        # appearance_weight, [N, R, num_group, R]
        appearance_weight = tf.transpose(scaled_dot, [0, 2, 1, 3])

        # relation_weight, [N, R, num_group, R]
        relation_weight = tf.log(tf.maximum(geometry_weight, 1e-6)) + appearance_weight
        relation_weight = tf.nn.softmax(relation_weight, axis=3)
        # [N, R*num_groups, R]
        relation_weight = tf.reshape(relation_weight, [N, R * self.num_groups, R])

        # [N, R*num_groups, input_size / num_groups]
        output = tf.matmul(relation_weight, dense_instances.get_field("value"))
        # [N, R, input_size]
        output = tf.reshape(output, [N, R, self.input_size])
        output = tf.gather_nd(output, instances.indices)

        return features + output