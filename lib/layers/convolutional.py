import tensorflow as tf
from .base import Layer
from ..utils import tf_utils
from ..utils import shape_utils
from ..utils.arg_scope import add_arg_scope

slim = tf.contrib.slim

__all__ = ["Conv2D", "DeformConv2D", "ModulatedDeformConv2D"]


def fix_padding(inputs, kernel_size, padding="SAME", rate=1):
    if padding == "SAME" and kernel_size != 1:
        kernel_size_effective = (
            kernel_size + (kernel_size - 1) * (rate - 1))
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(
            inputs,
            [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
        )
    return inputs


def _group_conv2d(inputs, weights, stride, rate, num_groups=1):
    stride = [1, stride, stride, 1]
    dilations = [1, rate, rate, 1]
    ret = None
    if tf_utils.get_tf_version_tuple() >= (1, 13):
        try:
            ret = tf.nn.conv2d(
                inputs, weights, stride, 'VALID', dilations=dilations
            )
        except ValueError:
            tf.logging.warn(
                "CUDNN group convolution is only available with"
                "https://github.com/tensorflow/tensorflow/pull/25818 ."
                "Will fall back to a loop-based slow implementation!")
    if ret is None:
        inputs = tf.split(inputs, num_groups, 3)
        kernels = tf.split(weights, num_groups, 3)
        outputs = [
            tf.nn.conv2d(i, k, stride, 'VALID', dilations=dilations)
            for i, k in zip(inputs, kernels)
        ]
        ret = tf.concat(outputs, 3)
    return ret


def _get_sampling_indices(grid_height, grid_width, kernel_size, stride, rate):
    # input feature map grid coordinates
    def ind2col(ind):
        ind = tf.reshape(ind, [1, *shape_utils.combined_static_and_dynamic_shape(ind), 1])
        ind = tf.image.extract_image_patches(
            ind,
            [1, kernel_size, kernel_size, 1],
            [1, stride, stride, 1],
            [1, rate, rate, 1],
            'VALID')
        return ind

    x, y = tf.meshgrid(tf.range(grid_width), tf.range(grid_height))
    # [grid_height, grid_width]
    y, x = ind2col(y), ind2col(x)
    # [1, out_h, out_w, kernel_size * kernel_size]
    return y, x


def _get_bilinear_interpolatied_pixels(inputs, y, x, in_h, in_w):
    # inputs: [batch_size, in_h, in_w,
    #          deform_num_groups, channels_per_group]
    # y, x: [batch_size, out_h, out_w, deform_num_groups,
    #        kernel_size * kernel_size]
    # get coordinates of points involved in interpolation
    y0, x0 = [tf.to_int32(tf.floor(i)) for i in [y, x]]
    y1, x1 = y0 + 1, x0 + 1
    y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
    x0, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x1]]

    # get pixel values involved in interpolation
    def get_pixel_values_at_points(inputs, y, x):
        batch, h, w, num_groups, n = shape_utils.combined_static_and_dynamic_shape(y)
        # y, x: [batch_size, out_h, out_w,
        #        deform_num_groups, kernel_size * kernel_size]

        batch_idx = tf.reshape(tf.range(batch), [batch, 1, 1, 1, 1])
        group_idx = tf.reshape(tf.range(num_groups), [1, 1, 1, num_groups, 1])
        b = tf.tile(batch_idx, [1, h, w, num_groups, n])
        g = tf.tile(group_idx, [batch, h, w, 1, n])
        pixel_idx = tf.stack([b, y, x, g], axis=-1)
        pixels = tf.gather_nd(inputs, pixel_idx)
        # pixels: [batch_size, out_h, out_w, deform_num_groups,
        #          kernel_size * kernel_size, channels_per_group]
        return pixels

    involved_indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
    p0, p1, p2, p3 = [get_pixel_values_at_points(inputs, y, x) for y, x in involved_indices]

    # bilinear interpolation kernel
    x0, x1, y0, y1 = [tf.to_float(i) for i in [x0, x1, y0, y1]]
    w0 = (y1 - y) * (x1 - x)
    w1 = (y1 - y) * (x - x0)
    w2 = (y - y0) * (x1 - x)
    w3 = (y - y0) * (x - x0)
    # bilinear interpolation
    # [batch_size, out_h, out_w, kernel_size * kernel_size,
    #  deform_num_groups, channels_per_group]
    w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]  # for broadcast
    # w: [batch_size, out_h, out_w, deform_num_groups,
    #     kernel_size * kernel_size, 1]
    pixels = [tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])]
    # pixels: [batch_size, out_h, out_w, deform_num_groups,
    #          kernel_size * kernel_size, channels_per_group]
    return pixels


@add_arg_scope
class Conv2D(Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding='SAME',
                 rate=1,
                 num_groups=1,
                 use_bias=True,
                 activation=None,
                 normalizer=None,
                 normalizer_params=None,
                 weights_initializer=None,
                 weights_regularizer=None,
                 bias_initializer=tf.zeros_initializer(),
                 bias_regularizer=None,
                 variables_collections=None,
                 trainable=True,
                 outputs_collections=None,
                 **kwargs):
        padding = padding.upper()
        if padding not in ['SAME', 'VALID']:
            raise ValueError('"padding" must be "SAME" or "VALID."')
        if in_channels % num_groups != 0:
            raise ValueError(
                '"in_channels" {:d} is not divisible by'
                ' "num_groups" {:d}.'.format(in_channels, num_groups))
        if out_channels % num_groups != 0:
            raise ValueError(
                '"out_channels" {:d} is not divisible by'
                ' "num_groups" {:d}.'.format(in_channels, num_groups))
        super(Conv2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            rate=rate,
            num_groups=num_groups,
            use_bias=use_bias,
            activation=activation,
            normalizer=normalizer,
            normalizer_params=normalizer_params,
            weights_initializer=weights_initializer,
            weights_regularizer=weights_regularizer,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            variables_collections=variables_collections,
            trainable=trainable,
            outputs_collections=outputs_collections,
            **kwargs)
        self.build()

    def build(self):
        filter_shape = [self.kernel_size, self.kernel_size,
                        self.in_channels / self.num_groups, self.out_channels]

        if self.weights_initializer is None:
            if tf_utils.get_tf_version_tuple() <= (1, 12):
                self.weights_initializer = (
                    tf.variance_scaling_initializer(scale=2.0))
            else:
                self.weights_initializer = (
                    tf.variance_scaling_initializer(
                        scale=2.0, distribution='untruncated_normal'))

        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:
            self.weights = slim.model_variable(
                "weights",
                shape=filter_shape,
                dtype=self.dtype,
                initializer=self.weights_initializer,
                regularizer=self.weights_regularizer,
                trainable=self.trainable,
                collections=self.variables_collections)

            if self.use_bias:
                if self.bias_initializer is None:
                    self.bias_initializer = tf.zeros_initializer()
                self.bias = slim.model_variable(
                    "bias",
                    shape=[self.out_channels],
                    dtype=self.dtype,
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    trainable=self.trainable,
                    collections=self.variables_collections)

            if self.normalizer is not None:
                normalizer_params = self.normalizer_params or {}
                normalizer_params = normalizer_params.copy()
                normalizer_params["channels"] = self.out_channels
                self.normalizer_fn = self.normalizer(**normalizer_params)

    def call(self, inputs):
        inputs = fix_padding(inputs, self.kernel_size, self.padding, self.rate)
        if self.num_groups == 1 and self.rate == 1:
            # slim.conv2d has bugs with dilations
            # (https://github.com/tensorflow/tensorflow/issues/26797)
            layer_variable_getter = self.build_variable_getter({'kernel': 'weights'})
            with tf.variable_scope(
                    self.sc, values=[inputs], reuse=True,
                    custom_getter=layer_variable_getter,
                    auxiliary_name_scope=False) as sc:
                layer = tf.layers.Conv2D(
                    filters=self.out_channels,
                    kernel_size=self.kernel_size,
                    strides=self.stride,
                    padding='VALID',
                    dilation_rate=self.rate,
                    activation=None,
                    use_bias=self.use_bias,
                    kernel_initializer=self.weights_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.weights_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    activity_regularizer=None,
                    trainable=self.trainable,
                    name=sc.name,
                    dtype=inputs.dtype.base_dtype,
                    _scope=sc,
                    _reuse=True)
                ret = layer.apply(inputs)
        else:
            ret = _group_conv2d(
                inputs,
                weights=self.weights,
                stride=self.stride,
                rate=self.rate,
                num_groups=self.num_groups
            )

            if self.use_bias:
                ret = tf.nn.bias_add(ret, self.bias)

        if self.normalizer is not None:
            ret = self.normalizer_fn(ret)

        if self.activation is not None:
            ret = self.activation(ret)

        return slim.utils.collect_named_outputs(
            self.outputs_collections, self.sc.name, ret)


@add_arg_scope
class DeformConv2D(Layer):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding='SAME',
        rate=1,
        num_groups=1,
        deform_num_groups=1,
        use_bias=True,
        activation=None,
        normalizer=None,
        normalizer_params=None,
        weights_initializer=None,
        weights_regularizer=None,
        bias_initializer=tf.zeros_initializer(),
        bias_regularizer=None,
        variables_collections=None,
        trainable=True,
        outputs_collections=None,
        **kwargs
    ):
        if in_channels % deform_num_groups != 0:
            raise ValueError(
                '"in_channels" {:d} is not divisible by "deform_num_groups"'
                ' {:d}.'.format(in_channels, deform_num_groups))
        super(DeformConv2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            rate=rate,
            num_groups=num_groups,
            deform_num_groups=deform_num_groups,
            use_bias=use_bias,
            activation=activation,
            normalizer=normalizer,
            normalizer_params=normalizer_params,
            weights_initializer=weights_initializer,
            weights_regularizer=weights_regularizer,
            bias_initializer=tf.zeros_initializer(),
            bias_regularizer=bias_regularizer,
            variables_collections=variables_collections,
            trainable=trainable,
            outputs_collections=outputs_collections,
            **kwargs)
        self.build()

    def build(self):
        filter_shape = [
            self.kernel_size, self.kernel_size,
            self.in_channels / self.num_groups, self.out_channels
        ]
        self.offset_channels = int(
            2 * self.deform_num_groups * self.kernel_size * self.kernel_size
        )
        offset_filter_shape = [
            self.kernel_size, self.kernel_size, self.in_channels, self.offset_channels
        ]

        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:
            self.weights = slim.model_variable(
                "weights",
                shape=filter_shape,
                dtype=self.dtype,
                initializer=self.weights_initializer,
                regularizer=self.weights_regularizer,
                trainable=self.trainable,
                collections=self.variables_collections)

            if self.use_bias:
                self.bias = slim.model_variable(
                    "bias",
                    shape=[self.out_channels],
                    dtype=self.dtype,
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    trainable=self.trainable,
                    collections=self.variables_collections)

            self.offset_weights = slim.model_variable(
                "offset_weights",
                shape=offset_filter_shape,
                dtype=self.dtype,
                initializer=tf.zeros_initializer(),
                regularizer=self.weights_regularizer,
                trainable=self.trainable,
                collections=self.variables_collections)

            self.offset_bias = slim.model_variable(
                "offset_bias",
                shape=[self.offset_channels],
                dtype=self.dtype,
                initializer=tf.zeros_initializer(),
                regularizer=self.bias_regularizer,
                trainable=self.trainable,
                collections=self.variables_collections)

            if self.normalizer is not None:
                normalizer_params = self.normalizer_params or {}
                if "channels" not in normalizer_params:
                    normalizer_params["channels"] = self.out_channels
                self.normalizer_fn = self.normalizer(**normalizer_params)

    def call(self, inputs):
        inputs = fix_padding(inputs, self.kernel_size, self.padding, self.rate)

        # slim.conv2d has bugs with dilations
        # (https://github.com/tensorflow/tensorflow/issues/26797)
        offset_layer_variable_getter = self.build_variable_getter({
            'bias': 'offset_bias',
            'kernel': 'offset_weights'
        })
        with tf.variable_scope(
                self.sc, values=[inputs], reuse=True,
                custom_getter=offset_layer_variable_getter,
                auxiliary_name_scope=False) as sc:
            layer = tf.layers.Conv2D(
                filters=self.offset_channels,
                kernel_size=self.kernel_size,
                strides=self.stride,
                padding='VALID',
                dilation_rate=self.rate,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.zeros_initializer,
                bias_initializer=tf.zeros_initializer,
                kernel_regularizer=self.weights_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=None,
                trainable=self.trainable,
                name=sc.name,
                dtype=inputs.dtype.base_dtype,
                _scope=sc,
                _reuse=True)
            offset = layer.apply(inputs)
        # [batch_size, out_h, out_w,
        #  kernel_size * kernel_size * deform_num_groups * 2]

        batch_size, in_h, in_w, in_channels = shape_utils.combined_static_and_dynamic_shape(inputs)
        out_h, out_w = shape_utils.combined_static_and_dynamic_shape(offset)[1:3]

        offset = tf.reshape(offset,
                            [batch_size, out_h, out_w, self.deform_num_groups,
                             self.kernel_size * self.kernel_size, 2])
        # [batch_size, out_h, out_w, deform_num_groups,
        #  kernel_size * kernel_size, 2]
        y_offset, x_offset = offset[..., 0], offset[..., 1]
        # [batch_size, out_h, out_w,
        #  deform_num_groups, kernel_size * kernel_size]

        # get original sampling positions
        y, x = _get_sampling_indices(in_h, in_w, self.kernel_size, self.stride, self.rate)
        # [1, out_h, out_w, kernel_size * kernel_size]

        # add offset
        y, x = [tf.expand_dims(i, axis=3) for i in [y, x]]  # for broadcast
        y = tf.cast(y, tf.float32) + y_offset
        x = tf.cast(x, tf.float32) + x_offset
        y = tf.clip_by_value(y, 0., tf.cast(in_h - 1, tf.float32))
        x = tf.clip_by_value(x, 0., tf.cast(in_w - 1, tf.float32))
        # [batch_size, out_h, out_w,
        #  deform_num_groups, kernel_size * kernel_size]

        channels_per_group = self.in_channels // self.deform_num_groups
        inputs = tf.reshape(inputs, [batch_size, in_h, in_w,
                                     self.deform_num_groups,
                                     channels_per_group])
        pixels = _get_bilinear_interpolatied_pixels(inputs, y, x, in_h, in_w)
        # pixels: [batch_size, out_h, out_w, deform_num_groups,
        #          kernel_size * kernel_size, channels_per_group]

        # reshape the pixel map
        pixels = tf.reshape(pixels, [batch_size, out_h, out_w,
                                     self.deform_num_groups,
                                     self.kernel_size, self.kernel_size,
                                     channels_per_group])
        pixels = tf.transpose(pixels, [0, 1, 4, 2, 5, 3, 6])
        pixels = tf.reshape(pixels, [batch_size,
                                     out_h * self.kernel_size,
                                     out_w * self.kernel_size,
                                     self.in_channels])
        # [batch_size, out_h * kernel_size, out_w * kernel_size, in_channels]

        if self.num_groups == 1 and self.rate == 1:
            # slim.conv2d has bugs with dilations
            # (https://github.com/tensorflow/tensorflow/issues/26797)
            layer_variable_getter = self.build_variable_getter({
                'kernel': 'weights'
            })
            with tf.variable_scope(
                    self.sc, values=[pixels], reuse=True,
                    custom_getter=layer_variable_getter,
                    auxiliary_name_scope=False) as sc:
                layer = tf.layers.Conv2D(
                    filters=self.out_channels,
                    kernel_size=self.kernel_size,
                    strides=self.kernel_size,
                    padding='VALID',
                    dilation_rate=1,
                    activation=None,
                    use_bias=self.use_bias,
                    kernel_initializer=self.weights_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.weights_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    activity_regularizer=None,
                    trainable=self.trainable,
                    name=sc.name,
                    dtype=pixels.dtype.base_dtype,
                    _scope=sc,
                    _reuse=True)
                ret = layer.apply(pixels)
        else:
            ret = _group_conv2d(
                pixels,
                weights=self.weights,
                stride=self.kernel_size,
                rate=1,
                num_groups=self.num_groups
            )

            if self.use_bias:
                ret = tf.nn.bias_add(ret, self.bias)

        if self.normalizer is not None:
            ret = self.normalizer_fn(ret)

        if self.activation is not None:
            ret = self.activation(ret)

        return slim.utils.collect_named_outputs(
            self.outputs_collections, self.sc.name, ret)


@add_arg_scope
class ModulatedDeformConv2D(Layer):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding='SAME',
        rate=1,
        num_groups=1,
        deform_num_groups=1,
        use_bias=True,
        activation=None,
        normalizer=None,
        normalizer_params=None,
        weights_initializer=None,
        weights_regularizer=None,
        bias_initializer=tf.zeros_initializer(),
        bias_regularizer=None,
        variables_collections=None,
        trainable=True,
        outputs_collections=None,
        **kwargs
    ):
        if in_channels % deform_num_groups != 0:
            raise ValueError(
                '"in_channels" {:d} is not divisible by "deform_num_groups"'
                ' {:d}.'.format(in_channels, deform_num_groups))
        super(DeformConv2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            rate=rate,
            num_groups=num_groups,
            deform_num_groups=deform_num_groups,
            use_bias=use_bias,
            activation=activation,
            normalizer=normalizer,
            normalizer_params=normalizer_params,
            weights_initializer=weights_initializer,
            weights_regularizer=weights_regularizer,
            bias_initializer=tf.zeros_initializer(),
            bias_regularizer=bias_regularizer,
            variables_collections=variables_collections,
            trainable=trainable,
            outputs_collections=outputs_collections,
            **kwargs)
        self.build()

    def build(self):
        filter_shape = [
            self.kernel_size, self.kernel_size,
            self.in_channels / self.num_groups, self.out_channels
        ]
        self.offset_channels = int(
            3 * self.kernel_size * self.deform_num_groups * self.kernel_size
        )
        offset_filter_shape = [
            self.kernel_size, self.kernel_size, self.in_channels, self.offset_channels
        ]

        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:
            self.weights = slim.model_variable(
                "weights",
                shape=filter_shape,
                dtype=self.dtype,
                initializer=self.weights_initializer,
                regularizer=self.weights_regularizer,
                trainable=self.trainable,
                collections=self.variables_collections)

            if self.use_bias:
                self.bias = slim.model_variable(
                    "bias",
                    shape=[self.out_channels],
                    dtype=self.dtype,
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    trainable=self.trainable,
                    collections=self.variables_collections)

            self.offset_weights = slim.model_variable(
                "offset_weights",
                shape=offset_filter_shape,
                dtype=self.dtype,
                initializer=tf.zeros_initializer(),
                regularizer=self.weights_regularizer,
                trainable=self.trainable,
                collections=self.variables_collections)

            self.offset_bias = slim.model_variable(
                "offset_bias",
                shape=[self.offset_channels],
                dtype=self.dtype,
                initializer=tf.zeros_initializer(),
                regularizer=self.bias_regularizer,
                trainable=self.trainable,
                collections=self.variables_collections)

            if self.normalizer is not None:
                normalizer_params = self.normalizer_params or {}
                if "channels" not in normalizer_params:
                    normalizer_params["channels"] = self.out_channels
                self.normalizer_fn = self.normalizer(**normalizer_params)

    def call(self, inputs):
        inputs = fix_padding(inputs, self.kernel_size, self.padding, self.rate)

        # slim.conv2d has bugs with dilations
        # (https://github.com/tensorflow/tensorflow/issues/26797)
        offset_layer_variable_getter = self.build_variable_getter({
            'bias': 'offset_bias',
            'kernel': 'offset_weights'
        })
        with tf.variable_scope(
                self.sc, values=[inputs], reuse=True,
                custom_getter=offset_layer_variable_getter,
                auxiliary_name_scope=False) as sc:
            layer = tf.layers.Conv2D(
                filters=self.offset_channels,
                kernel_size=self.kernel_size,
                strides=self.stride,
                padding='VALID',
                dilation_rate=self.rate,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.zeros_initializer,
                bias_initializer=tf.zeros_initializer,
                kernel_regularizer=self.weights_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=None,
                trainable=self.trainable,
                name=sc.name,
                dtype=inputs.dtype.base_dtype,
                _scope=sc,
                _reuse=True)
            offset = layer.apply(inputs)
        # [batch_size, out_h, out_w,
        #  kernel_size * kernel_size * deform_num_groups * 2]

        batch_size, in_h, in_w, in_channels = tf_utils.shape(inputs)
        out_h, out_w = shape_utils.combined_static_and_dynamic_shape(offset)[1:3]

        offset = tf.reshape(offset,
                            [batch_size, out_h, out_w, self.deform_num_groups,
                             self.kernel_size * self.kernel_size, 2])
        # [batch_size, out_h, out_w, deform_num_groups,
        #  kernel_size * kernel_size, 2]
        y_offset, x_offset = offset[..., 0], offset[..., 1]
        mod = tf.nn.sigmoid(offset[..., 2])
        # [batch_size, out_h, out_w,
        #  deform_num_groups, kernel_size * kernel_size]

        # get original sampling positions
        y, x = _get_sampling_indices(in_h, in_w, self.kernel_size, self.stride, self.rate)
        # [1, out_h, out_w, kernel_size * kernel_size]

        # add offset
        y, x = [tf.expand_dims(i, axis=3) for i in [y, x]]  # for broadcast
        y = tf.cast(y, tf.float32) + y_offset
        x = tf.cast(x, tf.float32) + x_offset
        y = tf.clip_by_value(y, 0., tf.cast(in_h - 1, tf.float32))
        x = tf.clip_by_value(x, 0., tf.cast(in_w - 1, tf.float32))
        # [batch_size, out_h, out_w,
        #  deform_num_groups, kernel_size * kernel_size]

        channels_per_group = self.in_channels // self.deform_num_groups
        inputs = tf.reshape(inputs, [batch, in_h, in_w,
                                     self.deform_num_groups,
                                     channels_per_group])
        pixels = _get_bilinear_interpolatied_pixels(inputs, y, x, in_h, in_w)
        pixels = pixels * tf.expand_dims(mod, axis=-1)
        # pixels: [batch_size, out_h, out_w, deform_num_groups,
        #          kernel_size * kernel_size, channels_per_group]

        # reshape the pixel map
        pixels = tf.reshape(pixels, [batch_size, out_h, out_w,
                                     self.deform_num_groups,
                                     self.kernel_size, self.kernel_size,
                                     channels_per_group])
        pixels = tf.transpose(pixels, [0, 1, 4, 2, 5, 3, 6])
        pixels = tf.reshape(pixels, [batch_size,
                                     out_h * self.kernel_size,
                                     out_w * self.kernel_size,
                                     self.in_channels])
        # [batch_size, out_h * kernel_size, out_w * kernel_size, in_channels]

        if self.num_groups == 1 and self.rate == 1:
            # slim.conv2d has bugs with dilations
            # (https://github.com/tensorflow/tensorflow/issues/26797)
            layer_variable_getter = self.build_variable_getter({
                'kernel': 'weights'
            })
            with tf.variable_scope(
                    self.sc, values=[pixels], reuse=True,
                    custom_getter=layer_variable_getter,
                    auxiliary_name_scope=False) as sc:
                layer = tf.layers.Conv2D(
                    filters=self.out_channels,
                    kernel_size=self.kernel_size,
                    strides=self.kernel_size,
                    padding='VALID',
                    dilation_rate=1,
                    activation=None,
                    use_bias=self.use_bias,
                    kernel_initializer=self.weights_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.weights_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    activity_regularizer=None,
                    trainable=self.trainable,
                    name=sc.name,
                    dtype=pixels.dtype.base_dtype,
                    _scope=sc,
                    _reuse=True)
                ret = layer.apply(pixels)
        else:
            ret = self.group_conv2d(pixels,
                                    weights=self.weights,
                                    stride=self.kernel_size,
                                    rate=1,
                                    num_groups=self.num_groups)

            if self.use_bias:
                ret = tf.nn.bias_add(ret, self.bias)

        if self.normalizer is not None:
            ret = self.normalizer_fn(ret)

        if self.activation is not None:
            ret = self.activation(ret)

        return slim.utils.collect_named_outputs(
            self.outputs_collections, self.sc.name, ret)


@add_arg_scope
class ConvTranspose2D(Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding='SAME',
                 use_bias=True,
                 activation=None,
                 normalizer=None,
                 normalizer_params=None,
                 weights_initializer=None,
                 weights_regularizer=None,
                 bias_initializer=tf.zeros_initializer(),
                 bias_regularizer=None,
                 variables_collections=None,
                 trainable=True,
                 outputs_collections=None,
                 **kwargs):
        padding = padding.upper()
        if padding not in ['SAME', 'VALID']:
            raise ValueError('"padding" must be "SAME" or "VALID."')
        super(ConvTranspose2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=use_bias,
            activation=activation,
            normalizer=normalizer,
            normalizer_params=normalizer_params,
            weights_initializer=weights_initializer,
            weights_regularizer=weights_regularizer,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            variables_collections=variables_collections,
            trainable=trainable,
            outputs_collections=outputs_collections,
            **kwargs)
        self.build()

    def build(self):
        filter_shape = [
            self.kernel_size, self.kernel_size, self.out_channels, self.in_channels
        ]

        if self.weights_initializer is None:
            if get_tf_version_tuple() <= (1, 12):
                self.weights_initializer = (
                    tf.variance_scaling_initializer(scale=2.0))
            else:
                self.weights_initializer = (
                    tf.variance_scaling_initializer(
                        scale=2.0, distribution='untruncated_normal'))

        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:
            self.weights = slim.model_variable(
                "weights",
                shape=filter_shape,
                dtype=self.dtype,
                initializer=self.weights_initializer,
                regularizer=self.weights_regularizer,
                trainable=self.trainable,
                collections=self.variables_collections)

            if self.use_bias:
                if self.bias_initializer is None:
                    self.bias_initializer = tf.zeros_initializer()
                self.bias = slim.model_variable(
                    "bias",
                    shape=[self.out_channels],
                    dtype=self.dtype,
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    trainable=self.trainable,
                    collections=self.variables_collections)

            if self.normalizer is not None:
                normalizer_params = self.normalizer_params or {}
                self.normalizer = normalizer(**normalizer_params)

    def call(self, inputs):
        if tf_utils.get_tf_version_tuple() <= (1, 12):
            layer_variable_getter = self.build_variable_getter({'kernel': 'weights'})
            with tf.variable_scope(
                    self.sc, values=[inputs], reuse=True,
                    custom_getter=layer_variable_getter,
                    auxiliary_name_scope=False) as sc:
                layer = tf.layers.Conv2DTranspose(
                    filters=self.out_channels,
                    kernel_size=self.kernel_size,
                    strides=self.stride,
                    padding=self.padding,
                    activation=None,
                    use_bias=self.use_bias,
                    kernel_initializer=self.weights_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.weights_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    activity_regularizer=None,
                    trainable=self.trainable,
                    name=sc.name,
                    dtype=inputs.dtype.base_dtype,
                    _scope=sc,
                    _reuse=True)
                ret = layer.apply(inputs)
        else:
            # Our own implementation, to avoid Keras bugs. 
            # https://github.com/tensorflow/tensorflow/issues/25946
            dynamic_shape = tf.shape(inputs)
            static_shape = inputs.shape.as_list()

            pad = 0
            if self.padding == "VALID":
                pad = max(self.kernel_size - self.stride, 0)

            dynamic_output_shape = [
                dynamic_shape[0],
                dynamic_shape[1] * self.stride + pad,
                dynamic_shape[2] * self.stride + pad,
                self.out_channels
            ]

            static_shape = [
                static_shape[0],
                None if static_shape[1] is None else static_shape[1] * self.stride + pad,
                None if static_shape[2] is None else static_shape[2] * self.stride + pad,
                self.out_channels
            ]
            ret = tf.nn.conv2d_transpose(
                inputs, self.weights, dynamic_output_shape,
                [1, self.stride, self.stride, 1], padding=self.padding
            )
            ret.set_shape(static_shape)
            if self.use_bias:
                ret = tf.nn.bias_add(ret, self.bias)

        if self.normalizer is not None:
            ret = self.normalizer(ret)

        if self.activation is not None:
            ret = self.activation(ret)

        return slim.utils.collect_named_outputs(
            self.outputs_collections, self.sc.name, ret)


class GCN(Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='SAME',
                 use_bias=True,
                 weights_initializer=None,
                 weights_regularizer=None,
                 bias_initializer=tf.zeros_initializer(),
                 bias_regularizer=None,
                 variables_collections=None,
                 trainable=True,
                 outputs_collections=None,
                 **kwargs):
        padding = padding.upper()
        if padding not in ['SAME', 'VALID']:
            raise ValueError('"padding" must be "SAME" or "VALID."')
        super(Conv2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            rate=rate,
            num_groups=num_groups,
            use_bias=use_bias,
            activation=activation,
            normalizer=normalizer,
            normalizer_params=normalizer_params,
            weights_initializer=weights_initializer,
            weights_regularizer=weights_regularizer,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            variables_collections=variables_collections,
            trainable=trainable,
            outputs_collections=outputs_collections,
            **kwargs)
        self.build()

    def build(self):
        filter_shape_1 = [
            self.kernel_size, 1, self.in_channels / self.num_groups, self.out_channels
        ]
        filter_shape_2 = [
            1, self.kernel_size, self.in_channels / self.num_groups, self.out_channels
        ]

        if self.weights_initializer is None:
            if get_tf_version_tuple() <= (1, 12):
                self.weights_initializer = (
                    tf.variance_scaling_initializer(scale=2.0))
            else:
                self.weights_initializer = (
                    tf.variance_scaling_initializer(
                        scale=2.0, distribution='untruncated_normal'))

        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as self.sc:
            self.l1_weights = slim.model_variable(
                "l1_weights",
                shape=filter_shape_1,
                dtype=self.dtype,
                initializer=self.weights_initializer,
                regularizer=self.weights_regularizer,
                trainable=self.trainable,
                collections=self.variables_collections)
            
            self.l2_weights = slim.model_variable(
                "l2_weights",
                shape=filter_shape_2,
                dtype=self.dtype,
                initializer=self.weights_initializer,
                regularizer=self.weights_regularizer,
                trainable=self.trainable,
                collections=self.variables_collections)

            self.r1_weights = slim.model_variable(
                "r1_weights",
                shape=filter_shape_1,
                dtype=self.dtype,
                initializer=self.weights_initializer,
                regularizer=self.weights_regularizer,
                trainable=self.trainable,
                collections=self.variables_collections)

            self.r2_weights = slim.model_variable(
                "r2_weights",
                shape=filter_shape_2,
                dtype=self.dtype,
                initializer=self.weights_initializer,
                regularizer=self.weights_regularizer,
                trainable=self.trainable,
                collections=self.variables_collections)

            if self.use_bias:
                if self.bias_initializer is None:
                    self.bias_initializer = tf.zeros_initializer()

                self.l1_bias = slim.model_variable(
                    "l1_bias",
                    shape=[self.out_channels],
                    dtype=self.dtype,
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    trainable=self.trainable,
                    collections=self.variables_collections)

                self.l2_bias = slim.model_variable(
                    "l2_bias",
                    shape=[self.out_channels],
                    dtype=self.dtype,
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    trainable=self.trainable,
                    collections=self.variables_collections)

                self.r1_bias = slim.model_variable(
                    "r1_bias",
                    shape=[self.out_channels],
                    dtype=self.dtype,
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    trainable=self.trainable,
                    collections=self.variables_collections)

                self.r2_bias = slim.model_variable(
                    "r2_bias",
                    shape=[self.out_channels],
                    dtype=self.dtype,
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    trainable=self.trainable,
                    collections=self.variables_collections)

    def call(self, inputs):

        ret_l = tf.nn.conv2d(
            inputs, self.l1_weights, strides=[1, 1, 1, 1],
            padding=[1, self.kernel_size, 1, 1], name="conv_l1"
        )
        if self.use_bias:
            ret_l = tf.nn.bias_add(ret_l, self.l1_bias)

        ret_l = tf.nn.conv2d(
            ret_l, self.l2_weights, strides=[1, 1, 1, 1],
            padding=[1, 1, self.kernel_size, 1], name="conv_l2"
        )
        if self.use_bias:
            ret_l = tf.nn.bias_add(ret_l, self.l1_bias)

        ret_r = tf.nn.conv2d(
            inputs, self.r1_weights, strides=[1, 1, 1, 1],
            padding=[1, self.kernel_size, 1, 1], name="conv_r1"
        )
        if self.use_bias:
            ret_r = tf.nn.bias_add(ret_r, self.r1_bias)

        ret_r = tf.nn.conv2d(
            ret_r, self.r2_weights, strides=[1, 1, 1, 1],
            padding=[1, self.kernel_size, 1, 1], name="conv_r2"
        )
        if self.use_bias:
            ret_r = tf.nn.bias_add(ret_r, self.r1_bias)

        ret = ret_l + ret_r

        return slim.utils.collect_named_outputs(
            self.outputs_collections, self.sc.name, ret)