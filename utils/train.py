import os
import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.data import data_import
from utils.nn import model, triplet_loss, random_mini_batches, save


def get_variables():
    """ 初始化 IO 变量 """
    A_in = tf.placeholder(dtype=tf.float32, shape=[None, 78, 30, 1], name="A_in")
    P_in = tf.placeholder(dtype=tf.float32, shape=[None, 78, 30, 1], name="P_in")
    N_in = tf.placeholder(dtype=tf.float32, shape=[None, 78, 30, 1], name="N_in")
    A_out = model(A_in)
    P_out = model(P_in)
    N_out = model(N_in)
    return (A_in, P_in, N_in), (A_out, P_out, N_out)

def train():
    """ 训练 """
    learning_rate = 0.0001
    num_epochs = 100
    mini_batch_size = 512
    GPU = True

    data_set = data_import()
    X_train_set = data_set["X_train_set"]
    X_dev_set = data_set["X_dev_set"]
    # X_dev_set = np.random.random_integers(0, 1, size = (3, len(data_set["X_dev_set"][0]), 78, 30, 1))
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
                train_permutation = list(np.random.permutation(train_size // 1000)) # 随机选取 1% 训练集数据用来训练
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
        save(sess)
