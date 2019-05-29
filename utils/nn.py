import os
import tensorflow as tf

from utils.config import Config
from utils.imager import H, W
from utils.data import data_import

GPU = True
if not GPU:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def conv2d(A, filter_size, num_filter, stride, padding='SAME', activation_function=lambda x: x):
    num_input_channels = int(A.shape[-1])
    W = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, num_input_channels, num_filter], stddev=0.1))
    tf.nn.conv2d(A, W, strides=[1,stride,stride,1], padding=padding)
    return activation_function(tf.nn.conv2d(A, W, strides=[1,stride,stride,1], padding=padding))

def max_pool(A, filter_size, stride, padding='SAME'):
    A = tf.nn.max_pool(A, ksize=[1,filter_size,filter_size,1], strides=[1,stride,stride,1], padding=padding)
    return A

def triplet_loss(anchor, positive, negative, alpha = 0.2):
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    return loss

def train():
    learning_rate = 0.0001
    num_epochs = 100

    data_set = data_import()
    X_train_set = data_set["X_train_set"]
    X_tag_train_set = data_set["X_tag_train_set"]
    X_dev_set = data_set["X_dev_set"]
    X_tag_dev_set = data_set["X_tag_dev_set"]
    A_in, P_in, N_in, A_out, P_out, N_out = nn()

    loss = triplet_loss(A_out, P_out, N_out, 0.2)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            _, train_cost = sess.run([train_step, loss], feed_dict={
                A_in: X_train_set[0][: 2000],
                P_in: X_train_set[1][: 2000],
                N_in: X_train_set[2][: 2000]
                })
            dev_cost  = sess.run(loss, feed_dict={
                A_in: X_dev_set[0][: 2000],
                P_in: X_dev_set[1][: 2000],
                N_in: X_dev_set[2][: 2000]
                })
            print("{} train cost is {} , dev cost is {}".format(
                epoch, sess.run(tf.reduce_mean(train_cost)), sess.run(tf.reduce_mean(dev_cost))))

def nn():
    A_in = tf.placeholder(dtype=tf.float32, shape=[None, 78, 30, 1])
    P_in = tf.placeholder(dtype=tf.float32, shape=[None, 78, 30, 1])
    N_in = tf.placeholder(dtype=tf.float32, shape=[None, 78, 30, 1])
    A_out = model(A_in)
    P_out = model(P_in)
    N_out = model(N_in)
    return A_in, P_in, N_in, A_out, P_out, N_out


def model(X):
    # 卷积 L1
    A1 = conv2d(X, 3, 8, 1, 'SAME', tf.nn.relu)
    A2 = max_pool(A1, 3, 3)
    # 卷积 L2
    A3 = conv2d(A2, 3, 16, 1, 'SAME', tf.nn.relu)
    A4 = max_pool(A3, 2, 2)
    # 转化为全连接层
    A5 = conv2d(A4, 5, 32, 4, 'VALID', tf.nn.relu)
    # 全连接
    # A6 = conv2d(A5, 1, 64, 1, 'VALID', tf.nn.relu)
    A6 = conv2d(A5, 1, 64, 4, 'VALID', tf.nn.relu)
    Y = conv2d(A6, 1, 128, 1, 'VALID', tf.nn.relu)
    return Y

