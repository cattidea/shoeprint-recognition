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


def conv2d(A, filter_size, num_filter, stride, num_layer, padding='SAME', activation_function=lambda x: x):
    """ 卷积操作 """
    num_input_channels = int(A.shape[-1])
    with tf.variable_scope("L" + str(num_layer), reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", shape=[filter_size[0], filter_size[1], num_input_channels, num_filter], dtype=tf.float32)
    # W = tf.Variable(tf.truncated_normal(shape=[filter_size[0], filter_size[1], num_input_channels, num_filter], stddev=0.1))
    tf.nn.conv2d(A, W, strides=[1,stride,stride,1], padding=padding)
    return activation_function(tf.nn.conv2d(A, W, strides=[1,stride,stride,1], padding=padding))

def max_pool(A, filter_size, stride, padding='SAME'):
    """ 最大池化操作 """
    A = tf.nn.max_pool(A, ksize=[1,filter_size[0],filter_size[1],1], strides=[1,stride,stride,1], padding=padding)
    return A

def random_mini_batches(data_set, mini_batch_size = 64, seed = 0):
    """ 切分训练集为 mini_batch """
    np.random.seed(seed)
    X_train_set = data_set["X_train_set"]
    train_size = len(X_train_set[0])
    permutation = list(np.random.permutation(train_size))
    batch_permutation_indices = [permutation[i: i + mini_batch_size] for i in range(0, train_size, mini_batch_size)]
    for batch_permutation in batch_permutation_indices:
        mini_batch_X = [X_train_set[0][batch_permutation], X_train_set[1][batch_permutation], X_train_set[2][batch_permutation]]
        yield mini_batch_X

def triplet_loss(anchor, positive, negative, alpha = 0.2):
    """ 计算三元组损失 """
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    return loss

def model(X):
    """ 神经网络模型 """
    # 卷积 L1
    A1 = conv2d(X, (3, 3), 8, 1, num_layer=1, padding='SAME', activation_function=tf.nn.relu)
    A2 = max_pool(A1, (3, 3), 3)
    # 卷积 L2
    A3 = conv2d(A2, (3, 3), 16, 1, num_layer=2, padding='SAME', activation_function=tf.nn.relu)
    A4 = max_pool(A3, (2, 2), 2)
    # 转化为全连接层
    A5 = conv2d(A4, (13, 5), 32, 4, num_layer=3, padding='VALID', activation_function=tf.nn.relu)
    # 全连接
    A6 = conv2d(A5, (1, 1), 64, 1, num_layer=4, padding='VALID', activation_function=tf.nn.relu)
    Y = conv2d(A6, (1, 1), 128, 1, num_layer=5, padding='VALID', activation_function=tf.nn.relu)
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
