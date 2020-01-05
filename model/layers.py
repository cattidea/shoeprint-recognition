import tensorflow as tf

def maxout(inputs, num_units, axis=None):
    """ 将前层部分参数作为 maxout 的参数进行处理 """
    shape = inputs.get_shape().as_list()
    if axis is None:
        # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not a multiple of num_units({})'
                            .format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    for i in range(len(shape)):
        if shape[i] is None:
            shape[i] = -1
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keepdims=False)
    return outputs


def conv2d(X, scope, filter, kernel_size=3, strides=1, padding="same", activation=None, batch_norm=False, is_training=False):
    """ 卷积，可添加 BN 层 """
    X = tf.layers.conv2d(inputs=X, filters=filter, kernel_size=kernel_size, strides=strides, padding=padding,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_CONV", reuse=tf.AUTO_REUSE)
    if batch_norm:
        X = tf.layers.batch_normalization(
            inputs=X, training=is_training, name=scope+"_BN", reuse=tf.AUTO_REUSE)
    if activation:
        X = activation(X)
    return X


def separable_conv2d(X, scope, filter, channel_multiplier=1, kernel_size=3, strides=1, padding="same", activation=None, batch_norm=False, is_training=False):
    """ 深度可分离卷积， 可添加 BN 层 """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        depthwish_filter = tf.get_variable(name='depthwish_filter', shape=[kernel_size[0], kernel_size[1], X.shape[3], channel_multiplier],
                                            initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
        pointwise_filter = tf.get_variable(name='pointwise_filter', shape=[1, 1, X.shape[3]*channel_multiplier, filter],
                                            initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
    X = tf.nn.separable_conv2d(input=X, depthwise_filter=depthwish_filter, pointwise_filter=pointwise_filter,
                                strides=[1, strides[0], strides[1], 1], padding=padding.upper())
    if batch_norm:
        X = tf.layers.batch_normalization(
            inputs=X, training=is_training, name=scope+"_BN", reuse=tf.AUTO_REUSE)
    if activation:
        X = activation(X)
    return X


def deepwise_conv2d(X, scope, channel_multiplier=1, kernel_size=3, strides=1, padding="same", activation=None, batch_norm=False, is_training=False):
    """ 逐层卷积，未添加 1 x 1 卷积，可添加 BN 层 """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        depthwise_filter = tf.get_variable(name='depthwise_filter', shape=[kernel_size[0], kernel_size[1], X.shape[3], channel_multiplier],
                                            initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
    X = tf.nn.depthwise_conv2d(input=X, filter=depthwise_filter, strides=[
                                1, strides[0], strides[1], 1], padding=padding.upper())
    if batch_norm:
        X = tf.layers.batch_normalization(
            inputs=X, training=is_training, name=scope+"_BN", reuse=tf.AUTO_REUSE)
    if activation:
        X = activation(X)
    return X


def dense(X, name, units, activation=None, keep_prob=1):
    """ 全连接层 """
    X = tf.layers.dense(inputs=X, units=units,
                        name=name, reuse=tf.AUTO_REUSE)
    if keep_prob != 1:
        X = tf.nn.dropout(X, keep_prob)
    if activation:
        X = activation(X)
    return X

def inception_v2(input, filters, scope, batch_norm=False, activation=None, is_training=False):
    """ inception_v2 网络
    ref: https://blog.csdn.net/loveliuzz/article/details/79135583
    """
    assert filters % 32 == 0
    k = filters // 32
    res_1_1 = tf.layers.conv2d(inputs=input, filters=8*k, kernel_size=1, strides=1, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_1_1", reuse=tf.AUTO_REUSE)
    res_1_1_t3 = tf.layers.conv2d(inputs=input, filters=12*k, kernel_size=1, strides=1, padding='same',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_1_1_t3", reuse=tf.AUTO_REUSE)
    res_3_3 = tf.layers.conv2d(inputs=res_1_1_t3, filters=16*k, kernel_size=3, strides=1, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_3_3", reuse=tf.AUTO_REUSE)
    res_1_1_t5 = tf.layers.conv2d(inputs=input, filters=k, kernel_size=1, strides=1, padding='same',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_1_1_t5", reuse=tf.AUTO_REUSE)
    res_3_3_t5 = tf.layers.conv2d(inputs=res_1_1_t5, filters=2*k, kernel_size=3, strides=1, padding='same',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_3_3_t5", reuse=tf.AUTO_REUSE)
    res_5_5 = tf.layers.conv2d(inputs=res_3_3_t5, filters=4*k, kernel_size=3, strides=1, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_5_5", reuse=tf.AUTO_REUSE)
    res_pool_t = tf.layers.max_pooling2d(
        input, pool_size=3, strides=1, padding='same')
    res_pool = tf.layers.conv2d(inputs=res_pool_t, filters=4*k, kernel_size=1, strides=1, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_pool", reuse=tf.AUTO_REUSE)
    res = tf.concat([res_1_1, res_3_3, res_5_5, res_pool], axis=-1)
    if batch_norm:
        res = tf.layers.batch_normalization(
            inputs=res, training=is_training, name=scope+"_BN", reuse=tf.AUTO_REUSE)
    if activation:
        res = activation(res)
    return res

def nnet_v1(input, filters, scope, strides=1, activate=True, batch_norm=False, is_training=False):
    """ 一个小尝试 """
    assert filters % 2 == 0
    X = input
    filters = filters * 2 if activate else filters

    X_dw_conv = conv2d(X, scope=scope+"_1_1_conv_1",
                            filter=filters//2, kernel_size=1, strides=1, padding="same")
    X_dw_conv = deepwise_conv2d(
        X_dw_conv, scope=scope+"_dw_conv_2", kernel_size=3, strides=1, padding="same")
    X_res = conv2d(X, scope=scope+"_1_1_conv_3", filter=filters //
                        2, kernel_size=1, strides=1, padding="same")
    X = tf.concat([X_dw_conv, X_res], axis=-1)

    if strides != 1:
        X_size_reduce_pool = conv2d(
            X, scope=scope+"_1_1_conv_4", filter=filters//2, kernel_size=1, strides=1, padding="same")
        X_size_reduce_pool = tf.layers.max_pooling2d(
            X_size_reduce_pool, pool_size=3, strides=strides, padding="same")
        X_size_reduce_dw_conv = conv2d(
            X, scope=scope+"_1_1_conv_5", filter=filters//2, kernel_size=1, strides=1, padding="same")
        X_size_reduce_dw_conv = deepwise_conv2d(
            X_size_reduce_dw_conv, scope=scope+"_size_reduce_conv_6", kernel_size=3, strides=strides, padding="same")
        X = tf.concat([X_size_reduce_pool, X_size_reduce_dw_conv], axis=-1)

    if activate:
        X_activate = conv2d(
            X, scope=scope+"_1_1_conv_7", filter=filters//4*3, kernel_size=1, strides=1, padding="same")
        X_activate = maxout(inputs=X_activate, num_units=filters//4)
        X_no_activate = conv2d(
            X, scope=scope+"_1_1_conv_8", filter=filters//4, kernel_size=1, strides=1, padding="same")
        X = tf.concat([X_activate, X_no_activate], axis=-1)

    if batch_norm:
        X = tf.layers.batch_normalization(
            inputs=X, training=is_training, name=scope+"_BN", reuse=tf.AUTO_REUSE)

    return X
