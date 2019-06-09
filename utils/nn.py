import os
import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.imager import H, W
from utils.data import data_import


CONFIG = Config()
MODEL_PATH = CONFIG['model_path']
MODEL_DIR = CONFIG['model_dir']
MODEL_META = CONFIG['model_meta']


def add_layer(A, out_size, num_layer, activation_function=lambda x: x,
              keep_prob=1, batch_norm=True, is_training=True):
    in_size = int(A.shape[-1])
    with tf.variable_scope("L" + str(num_layer), reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", shape=[in_size, out_size], initializer=tf.random_normal_initializer(stddev=0.1, dtype=tf.float32))
        b = tf.get_variable("b", shape=[1, out_size], initializer=tf.random_normal_initializer(stddev=0.1, dtype=tf.float32))
        print(W.name, b.name)
    Wx_plus_b = tf.matmul(A, W) + b
    if keep_prob != 1:
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if batch_norm:
        Wx_plus_b = batch_norm_layer(Wx_plus_b, is_training=is_training)
    return activation_function(Wx_plus_b)

def conv2d(A, filter_size, num_filter, stride, num_layer, padding='SAME',
           activation_function=lambda x: x, batch_norm=True, is_training=True):
    """ 卷积操作 """
    num_input_channels = int(A.shape[-1])
    with tf.variable_scope("L" + str(num_layer), reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", shape=[filter_size[0], filter_size[1], num_input_channels, num_filter],
                                 initializer=tf.random_normal_initializer(stddev=0.1, dtype=tf.float32))
        print(W.name)
    Wa = tf.nn.conv2d(A, W, strides=[1,stride,stride,1], padding=padding)
    if batch_norm:
        Wa = batch_norm_layer(Wa, is_training=is_training)
    return activation_function(Wa)

def max_pool(A, filter_size, stride, padding='SAME'):
    """ 最大池化操作 """
    A = tf.nn.max_pool(A, ksize=[1,filter_size[0],filter_size[1],1], strides=[1,stride,stride,1], padding=padding)
    return A

def batch_norm_layer(A, is_training=False):
    """ 批量归一化 """
    return tf.contrib.layers.batch_norm(inputs=A, decay=0.9, updates_collections=None, is_training=is_training)

def flatten(A):
    return tf.contrib.layers.flatten(A)

def random_mini_batches(data_set, mini_batch_size = 64, seed = 0):
    """ 切分训练集为 mini_batch """
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

def model(X, is_training, keep_prob):
    """ 神经网络模型 """
    # # 卷积 L1
    # A1 = conv2d(X, (3, 3), 8, 1, num_layer=1, padding='SAME', activation_function=tf.nn.relu, batch_norm=True, is_training=is_training)
    # A2 = max_pool(A1, (3, 3), 3)
    # # 卷积 L2
    # A3 = conv2d(A2, (3, 3), 16, 1, num_layer=2, padding='SAME', activation_function=tf.nn.relu, batch_norm=True, is_training=is_training)
    # A4 = max_pool(A3, (3, 3), 2)

    # """ V1 """
    # # 转化为全连接层
    # # A5 = conv2d(A4, (13, 5), 32, 4, num_layer=3, padding='VALID', activation_function=tf.nn.relu, batch_norm=True, is_training=is_training)
    # # 全连接
    # # A6 = conv2d(A5, (1, 1), 64, 1, num_layer=4, padding='VALID', activation_function=tf.nn.relu, batch_norm=True, is_training=is_training)
    # # Y = conv2d(A6, (1, 1), 128, 1, num_layer=5, padding='VALID', activation_function=tf.nn.relu, batch_norm=True, is_training=is_training)

    # """ V2 """
    # # 卷积 L3
    # A5 = conv2d(A4, (5, 5), 32, 4, num_layer=3, padding='VALID', activation_function=tf.nn.relu, batch_norm=True, is_training=is_training)
    # # 转化为全连接层
    # A6 = flatten(A5)
    # # 全连接
    # A7 = add_layer(A6, 128, num_layer=4, activation_function=tf.nn.relu, keep_prob=keep_prob, batch_norm=True, is_training=is_training)
    # Y = add_layer(A7, 64, num_layer=5, activation_function=tf.nn.relu, keep_prob=keep_prob, batch_norm=True, is_training=is_training)
    # # Y = tf.contrib.layers.fully_connected(A6, 64, activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE, scope="L000") # 简易版全连接测试

    A0 = X
    # A0 = tf.layers.batch_normalization(inputs=A0, training=is_training, name="BN0", reuse=tf.AUTO_REUSE)
    print("A0: {}".format(A0.shape))

    # CONV L1
    A1 = tf.layers.conv2d(inputs=A0, filters=8, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV1", reuse=tf.AUTO_REUSE)
    A1 = tf.layers.batch_normalization(inputs=A1, training=is_training, name="BN1", reuse=tf.AUTO_REUSE)
    A1 = tf.nn.relu(A1)
    A1 = tf.layers.max_pooling2d(A1, pool_size=3, strides=2, padding='same')
    print("A1: {}".format(A1.shape))

    # CONV L2
    A2 = tf.layers.conv2d(inputs=A1, filters=16, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV2", reuse=tf.AUTO_REUSE)
    A2 = tf.layers.batch_normalization(inputs=A2, training=is_training, name="BN2", reuse=tf.AUTO_REUSE)
    A2 = tf.nn.relu(A2)
    A2 = tf.layers.max_pooling2d(A2, pool_size=3, strides=2, padding='same')
    print("A2: {}".format(A2.shape))

    # CONV L3
    A3 = tf.layers.conv2d(inputs=A2, filters=32, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV3", reuse=tf.AUTO_REUSE)
    A3 = tf.layers.batch_normalization(inputs=A3, training=is_training, name="BN3", reuse=tf.AUTO_REUSE)
    A3 = tf.nn.relu(A3)
    A3 = tf.layers.max_pooling2d(A3, pool_size=3, strides=2, padding='same')
    print("A3: {}".format(A3.shape))

    # CONV L4
    A4 = tf.layers.conv2d(inputs=A3, filters=64, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV4", reuse=tf.AUTO_REUSE)
    A4 = tf.layers.batch_normalization(inputs=A4, training=is_training, name="BN4", reuse=tf.AUTO_REUSE)
    A4 = tf.nn.relu(A4)
    A4 = tf.layers.max_pooling2d(A4, pool_size=3, strides=2, padding='same')
    print("A4: {}".format(A4.shape))

    # CONV L5
    A5 = tf.layers.conv2d(inputs=A4, filters=128, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV5", reuse=tf.AUTO_REUSE)
    A5 = tf.layers.batch_normalization(inputs=A5, training=is_training, name="BN5", reuse=tf.AUTO_REUSE)
    A5 = tf.nn.relu(A5)
    A5 = tf.layers.max_pooling2d(A5, pool_size=5, strides=4, padding='valid')
    print("A5: {}".format(A5.shape))

    # flatten
    A6 = tf.layers.flatten(A5)
    print("A6: {}".format(A6.shape))

    # FC L1
    A7 = tf.layers.dense(inputs=A6, units=512, name="FC1", reuse=tf.AUTO_REUSE)
    # A7 = tf.layers.batch_normalization(inputs=A7, training=is_training, name="BN6", reuse=tf.AUTO_REUSE)
    if keep_prob != 1:
        A7 = tf.nn.dropout(A7, keep_prob)
    A7 = tf.nn.relu(A7)
    print("A7: {}".format(A7.shape))

    # FC L2
    A8 = tf.layers.dense(inputs=A7, units=512, name="FC2", reuse=tf.AUTO_REUSE)
    # A8 = tf.layers.batch_normalization(inputs=A8, training=is_training, name="BN7", reuse=tf.AUTO_REUSE)
    if keep_prob != 1:
        A8 = tf.nn.dropout(A8, keep_prob)
    A8 = tf.nn.relu(A8)
    print("A8: {}".format(A8.shape))

    # FC L3
    A9 = tf.layers.dense(inputs=A8, units=128, name="FC3", reuse=tf.AUTO_REUSE)
    # A9 = tf.layers.batch_normalization(inputs=A9, training=is_training, name="BN8", reuse=tf.AUTO_REUSE)
    if keep_prob != 1:
        A9 = tf.nn.dropout(A9, keep_prob)
    A9 = tf.nn.relu(A9)
    print("A9: {}".format(A9.shape))


    Y = A7
    # Y = tf.layers.batch_normalization(inputs=Y, training=is_training, name="BN7", reuse=tf.AUTO_REUSE)
    return Y

def save(sess):
    saver=tf.train.Saver(max_to_keep=1)
    saver.save(sess, MODEL_PATH)

def restore():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(MODEL_META)
    saver = tf.train.Saver(max_to_keep=1)
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
    return sess
