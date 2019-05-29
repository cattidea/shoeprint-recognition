import os
import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.imager import H, W
from utils.data import data_import


CONFIG = Config()
MODEL_PATH = CONFIG['model_path']
MODEL_PATH = CONFIG['model_dir']


def conv2d(A, filter_size, num_filter, stride, padding='SAME', activation_function=lambda x: x):
    """ 卷积操作 """
    num_input_channels = int(A.shape[-1])
    W = tf.Variable(tf.truncated_normal(shape=[filter_size[0], filter_size[1], num_input_channels, num_filter], stddev=0.1))
    tf.nn.conv2d(A, W, strides=[1,stride,stride,1], padding=padding)
    return activation_function(tf.nn.conv2d(A, W, strides=[1,stride,stride,1], padding=padding))

def max_pool(A, filter_size, stride, padding='SAME'):
    """ 最大池化操作 """
    A = tf.nn.max_pool(A, ksize=[1,filter_size[0],filter_size[1],1], strides=[1,stride,stride,1], padding=padding)
    return A

def random_mini_batches(data_set, mini_batch_size = 64, seed = 0):
    """ 切分为 mini_batch """
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

def train():
    """ 训练 """
    learning_rate = 0.0001
    num_epochs = 5
    mini_batch_size = 512
    GPU = True

    data_set = data_import()
    X_train_set = data_set["X_train_set"]
    X_dev_set = data_set["X_dev_set"]
    train_size = len(X_train_set[0])
    dev_size = len(X_dev_set[0])

    (A_in, P_in, N_in), (A_out, P_out, N_out) = get_variables()
    loss = triplet_loss(A_out, P_out, N_out, 0.2)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)


    if GPU:
        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            if not mini_batch_size:
                train_permutation = list(np.random.permutation(train_size // 100)) # 随机选取 1% 训练集数据用来训练
                _, train_cost = sess.run([train_step, loss], feed_dict={
                    A_in: X_train_set[0][train_permutation],
                    P_in: X_train_set[1][train_permutation],
                    N_in: X_train_set[2][train_permutation]
                    })
            else:
                train_cost = 0
                for X_mini_batch in random_mini_batches(data_set, mini_batch_size=mini_batch_size):
                    _, temp_cost = sess.run([train_step, loss], feed_dict={
                        A_in: X_mini_batch[0],
                        P_in: X_mini_batch[1],
                        N_in: X_mini_batch[2]
                        })
                    train_cost += temp_cost / (train_size // mini_batch_size)
            dev_permutation = list(np.random.permutation(dev_size // 10)) # 随机选取 10% 开发集数据用来测试
            dev_cost  = sess.run(loss, feed_dict={
                A_in: X_dev_set[0][dev_permutation],
                P_in: X_dev_set[1][dev_permutation],
                N_in: X_dev_set[2][dev_permutation]
                })
            print("{} train cost is {} , dev cost is {}".format(
                epoch, sess.run(tf.reduce_mean(train_cost)), sess.run(tf.reduce_mean(dev_cost))))

def get_variables():
    """ 获取 IO 变量 """
    A_in = tf.placeholder(dtype=tf.float32, shape=[None, 78, 30, 1])
    P_in = tf.placeholder(dtype=tf.float32, shape=[None, 78, 30, 1])
    N_in = tf.placeholder(dtype=tf.float32, shape=[None, 78, 30, 1])
    A_out = model(A_in)
    P_out = model(P_in)
    N_out = model(N_in)
    return (A_in, P_in, N_in), (A_out, P_out, N_out)


def model(X):
    """ 神经网络模型 """
    # 卷积 L1
    A1 = conv2d(X, (3, 3), 8, 1, 'SAME', tf.nn.relu)
    A2 = max_pool(A1, (3, 3), 3)
    # 卷积 L2
    A3 = conv2d(A2, (3, 3), 16, 1, 'SAME', tf.nn.relu)
    A4 = max_pool(A3, (2, 2), 2)
    # 转化为全连接层
    A5 = conv2d(A4, (13, 5), 32, 4, 'VALID', tf.nn.relu)
    # 全连接
    A6 = conv2d(A5, (1, 1), 64, 1, 'VALID', tf.nn.relu)
    Y = conv2d(A6, (1, 1), 128, 1, 'VALID', tf.nn.relu)
    return Y

def save(sess):
    saver=tf.train.Saver(max_to_keep=1)
    saver.save(sess, MODEL_PATH)

def restore():
    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=1)
    model_file = tf.train.latest_checkpoint(MODEL_DIR)
    saver.restore(sess, model_file)
    return sess
