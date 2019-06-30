import os
import numpy as np
import tensorflow as tf

from utils.imager import H, W


def random_mini_batches(data_set, mini_batch_size = 64, seed = 0):
    """ 随机切分训练集为 mini_batch """
    np.random.seed(seed)
    data_size = len(data_set[0])
    permutation = list(np.random.permutation(data_size))
    batch_permutation_indices = [permutation[i: i + mini_batch_size] for i in range(0, data_size, mini_batch_size)]
    for batch_permutation in batch_permutation_indices:
        mini_batch_X = [data_set[0][batch_permutation], data_set[1][batch_permutation], data_set[2][batch_permutation]]
        yield mini_batch_X

def triplet_loss(anchor, positive, negative, alpha = 0.2):
    """ 计算三元组损失 """
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    return loss

def compute_embeddings(input, masks, sess, ops, step=512):
    array_length = len(input)
    embeddings = np.zeros((array_length, *ops["embeddings"].shape[1: ]), dtype=np.float32)

    for i in range(0, array_length, step):
        input_batch = input[i: i + step]
        masks_batch = masks[i: i + step]
        embeddings_batch = sess.run(ops["embeddings"], feed_dict={
            ops["input"]: (input_batch / 255).astype(np.float32),
            ops["masks"]: masks_batch.astype(np.float32),
            ops["is_training"]: False,
            ops["keep_prob"]: 1
            })
        embeddings[i: i+step] = embeddings_batch
    return embeddings


def maxout_layer(input, k, scope):
    """ 新增参数的 maxout 层
    ref: https://www.jianshu.com/p/710fd5d6d640
    """
    num_channels = input.get_shape().as_list()[-1]
    assert num_channels % k == 0
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", shape=(num_channels, num_channels//k, k), initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", shape=(num_channels//k, k), initializer=tf.random_normal_initializer(stddev=0.1))
    return tf.reduce_max(tf.tensordot(input, W, axes=1) + b, axis=2)


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


def inception_v2(input, filters, scope, is_training):
    """ inception_v2 网络
    ref: https://blog.csdn.net/loveliuzz/article/details/79135583
    """
    assert filters % 32 == 0
    k = filters // 32
    # input = tf.layers.batch_normalization(inputs=input, training=is_training, name=scope+"_BN", reuse=tf.AUTO_REUSE)
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
    res_pool_t = tf.layers.max_pooling2d(input, pool_size=3, strides=1, padding='same')
    res_pool = tf.layers.conv2d(inputs=res_pool_t, filters=4*k, kernel_size=1, strides=1, padding='same',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_pool", reuse=tf.AUTO_REUSE)
    res = tf.concat([res_1_1, res_3_3, res_5_5, res_pool], axis=-1)
    res = tf.nn.relu(res)
    return res


def model_132_48_base(X, is_training, keep_prob):
    A0 = X
    # A0 = tf.layers.batch_normalization(inputs=A0, training=is_training, name="BN0", reuse=tf.AUTO_REUSE)
    print("A0: {}".format(A0.shape))

    # CONV L1
    A1 = tf.layers.conv2d(inputs=A0, filters=8, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV1", reuse=tf.AUTO_REUSE)
    # A1 = tf.layers.batch_normalization(inputs=A1, training=is_training, name="BN1", reuse=tf.AUTO_REUSE)
    A1 = tf.nn.relu(A1)
    A1 = tf.layers.max_pooling2d(A1, pool_size=2, strides=2)
    print("A1: {}".format(A1.shape))

    # CONV L2
    A2 = tf.layers.conv2d(inputs=A1, filters=16, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV2", reuse=tf.AUTO_REUSE)
    # A2 = tf.layers.batch_normalization(inputs=A2, training=is_training, name="BN2", reuse=tf.AUTO_REUSE)
    A2 = tf.nn.relu(A2)
    A2 = tf.layers.max_pooling2d(A2, pool_size=2, strides=2)
    print("A2: {}".format(A2.shape))

    # CONV L3
    A3 = tf.layers.conv2d(inputs=A2, filters=32, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV3", reuse=tf.AUTO_REUSE)
    # A3 = tf.layers.batch_normalization(inputs=A3, training=is_training, name="BN3", reuse=tf.AUTO_REUSE)
    A3 = tf.nn.relu(A3)
    A3 = tf.layers.max_pooling2d(A3, pool_size=3, strides=3)
    print("A3: {}".format(A3.shape))

    # flatten
    A4 = tf.layers.flatten(A3)
    print("A4: {}".format(A4.shape))

    # FC L1
    A5 = tf.layers.dense(inputs=A4, units=1024, name="FC1", reuse=tf.AUTO_REUSE)
    if keep_prob != 1:
        A5 = tf.nn.dropout(A5, keep_prob)
    A5 = maxout(A5, num_units=512)
    print("A5: {}".format(A5.shape))

    # FC L2
    A6 = tf.layers.dense(inputs=A5, units=512, name="FC2", reuse=tf.AUTO_REUSE)
    if keep_prob != 1:
        A6 = tf.nn.dropout(A6, keep_prob)
    A6 = maxout(A6, num_units=256)
    print("A6: {}".format(A6.shape))

    # FC L3
    A7 = tf.layers.dense(inputs=A6, units=256, name="FC3", reuse=tf.AUTO_REUSE)
    if keep_prob != 1:
        A7 = tf.nn.dropout(A7, keep_prob)
    A7 = maxout(A7, num_units=128)
    print("A7: {}".format(A7.shape))

    Y = A7
    Y = tf.nn.l2_normalize(Y, axis=-1)
    print("Y: {}".format(Y.shape))

    return Y


def model_132_48_v2(X, is_training, keep_prob):
    A0 = X
    # A0 = tf.layers.batch_normalization(inputs=A0, training=is_training, name="BN0", reuse=tf.AUTO_REUSE)
    print("A0: {}".format(A0.shape))

    # CONV L1
    A1 = tf.layers.conv2d(inputs=A0, filters=8, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV1", reuse=tf.AUTO_REUSE)
    # A1 = tf.layers.batch_normalization(inputs=A1, training=is_training, name="BN1", reuse=tf.AUTO_REUSE)
    A1 = tf.nn.relu(A1)
    A1 = tf.layers.max_pooling2d(A1, pool_size=3, strides=3)
    print("A1: {}".format(A1.shape))

    # CONV L2
    A2 = tf.layers.conv2d(inputs=A1, filters=16, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV2", reuse=tf.AUTO_REUSE)
    # A2 = tf.layers.batch_normalization(inputs=A2, training=is_training, name="BN2", reuse=tf.AUTO_REUSE)
    A2 = tf.nn.relu(A2)
    A2 = tf.layers.max_pooling2d(A2, pool_size=2, strides=2)
    print("A2: {}".format(A2.shape))

    # CONV L3
    A3 = tf.layers.conv2d(inputs=A2, filters=64, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV3", reuse=tf.AUTO_REUSE)
    # A3 = tf.layers.batch_normalization(inputs=A3, training=is_training, name="BN3", reuse=tf.AUTO_REUSE)
    A3 = tf.nn.relu(A3)
    A3 = tf.layers.max_pooling2d(A3, pool_size=2, strides=2)
    print("A3: {}".format(A3.shape))

    # flatten
    A4 = tf.layers.flatten(A3)
    print("A4: {}".format(A4.shape))

    # FC L1
    A5 = tf.layers.dense(inputs=A4, units=2048, name="FC1", reuse=tf.AUTO_REUSE)
    if keep_prob != 1:
        A5 = tf.nn.dropout(A5, keep_prob)
    A5 = maxout(A5, num_units=512)
    print("A5: {}".format(A5.shape))

    # FC L2
    A6 = tf.layers.dense(inputs=A5, units=512, name="FC2", reuse=tf.AUTO_REUSE)
    if keep_prob != 1:
        A6 = tf.nn.dropout(A6, keep_prob)
    A6 = maxout(A6, num_units=256)
    print("A6: {}".format(A6.shape))

    # FC L3
    A7 = tf.layers.dense(inputs=A6, units=256, name="FC3", reuse=tf.AUTO_REUSE)
    if keep_prob != 1:
        A7 = tf.nn.dropout(A7, keep_prob)
    A7 = maxout(A7, num_units=128)
    print("A7: {}".format(A7.shape))

    Y = A7
    Y = tf.nn.l2_normalize(Y, axis=-1)
    print("Y: {}".format(Y.shape))

    return Y


def model_132_48_inception(X, mask, is_training, keep_prob):
    A0 = X
    # A0 = tf.layers.batch_normalization(inputs=A0, training=is_training, name="BN0", reuse=tf.AUTO_REUSE)
    print("A0: {}".format(A0.shape))

    # CONV L1
    A1 = tf.layers.conv2d(inputs=A0, filters=8, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV1", reuse=tf.AUTO_REUSE)
    # A1 = tf.layers.batch_normalization(inputs=A1, training=is_training, name="BN1", reuse=tf.AUTO_REUSE)
    A1 = tf.nn.relu(A1)
    A1 = tf.layers.max_pooling2d(A1, pool_size=3, strides=3)
    print("A1: {}".format(A1.shape))

    # CONV L2
    A2 = tf.layers.conv2d(inputs=A1, filters=16, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV2", reuse=tf.AUTO_REUSE)
    # A2 = tf.layers.batch_normalization(inputs=A2, training=is_training, name="BN2", reuse=tf.AUTO_REUSE)
    A2 = tf.nn.relu(A2)
    A2 = tf.layers.max_pooling2d(A2, pool_size=2, strides=2)
    print("A2: {}".format(A2.shape))

    # INCEPTION L1
    A3 = inception_v2(input=A2, scope="INCEPTION1", filters=64, is_training=is_training)
    print("A3: {}".format(A3.shape))

    # INCEPTION L2
    A4 = inception_v2(input=A3, scope="INCEPTION2", filters=128, is_training=is_training)
    print("A4: {}".format(A4.shape))

    # CONV to 1
    A5 = A4 * mask
    A5 = tf.layers.conv2d(inputs=A5, filters=256, kernel_size=(7, 5), strides=3, padding='valid',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV_to_6_2", reuse=tf.AUTO_REUSE)
    A5 = tf.layers.conv2d(inputs=A5, filters=512, kernel_size=(6, 2), strides=1, padding='valid',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV_to_1_1", reuse=tf.AUTO_REUSE)
    # A5 = tf.nn.relu(A5)
    A5 = maxout(A5, num_units=128)
    print("A5: {}".format(A5.shape))

    # flatten
    A6 = tf.layers.flatten(A5)
    print("A6: {}".format(A6.shape))

    Y = A6
    Y = tf.nn.l2_normalize(Y, axis=-1)
    print("Y: {}".format(Y.shape))

    return Y


def model(X, mask, is_training, keep_prob):
    """ 神经网络模型 """

    return model_132_48_inception(X, mask, is_training, keep_prob)
