import os
import time
import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.data import data_import
from utils.nn import model, triplet_loss, nucleus_loss, random_mini_batches, save
from utils.imager import H as IH, W as IW

RADIUS = 0.05
MARGIN = 4 * RADIUS

def get_variables():
    """ 初始化 IO 变量 """
    A_in = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="A_in")
    P_in = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="P_in")
    N_in = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="N_in")
    is_training = tf.placeholder(dtype=tf.bool, name="is_training")
    keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    A_out = model(A_in, is_training, keep_prob)
    P_out = model(P_in, is_training, keep_prob)
    N_out = model(N_in, is_training, keep_prob)
    return (A_in, P_in, N_in), (A_out, P_out, N_out), is_training, keep_prob

def train():
    """ 训练 """
    learning_rate = 0.0001
    num_epochs = 8
    mini_batch_size = 128
    amplify = 3
    GPU = True

    # load data
    data_set = data_import(amplify=amplify)
    X_indices_train_set = data_set["X_indices_train_set"]
    X_indices_dev_set = data_set["X_indices_dev_set"]
    X_simple_indices = data_set["X_simple_indices"]
    X_imgs = data_set["X_imgs"]
    train_size = len(X_indices_train_set[0])
    dev_size = len(X_indices_dev_set[0])

    # GPU Config
    config = tf.ConfigProto(allow_soft_placement=True)
    if GPU:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # train 计算图
    (A_in, P_in, N_in), (A_out, P_out, N_out), is_training, keep_prob = get_variables()
    print(A_in.name, A_out.name, is_training.name, keep_prob.name)
    print(P_in.name, P_out.name, is_training.name, keep_prob.name)
    print(N_in.name, N_out.name, is_training.name, keep_prob.name)
    loss = triplet_loss(A_out, P_out, N_out, MARGIN)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    # dist-train 计算图
    dist_loss = nucleus_loss(A_out, P_out, 10*RADIUS)
    dist_optimizer = tf.train.AdamOptimizer(learning_rate)
    dist_train_step = dist_optimizer.minimize(dist_loss)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            # train
            if not mini_batch_size:
                train_batch_size = train_size // 10000 # 随机选取 0.01% 训练集数据用来训练
                train_permutation = list(np.random.permutation(train_batch_size))
                _, train_cost = sess.run([train_step, loss], feed_dict={
                    A_in: X_imgs[X_indices_train_set[0][train_permutation]],
                    P_in: X_imgs[X_indices_train_set[1][train_permutation]],
                    N_in: X_imgs[X_indices_train_set[2][train_permutation]],
                    is_training: True,
                    keep_prob: 0.5
                    })
                train_cost /= train_batch_size
            else:
                # train
                train_cost = 0
                for batch_index, (X_indices_mini_batch, mini_batch_simple_indices) in enumerate(
                    random_mini_batches(X_indices_train_set, X_simple_indices, mini_batch_size=mini_batch_size)):
                    # dist-train
                    if epoch in [0, 2]:
                        _, temp_dist_cost = sess.run([dist_train_step, dist_loss], feed_dict={
                            A_in: X_imgs[mini_batch_simple_indices[0]],
                            P_in: X_imgs[mini_batch_simple_indices[1]],
                            is_training: True,
                            keep_prob: 0.5
                            })
                    else:
                        temp_dist_cost = -1

                    _, temp_cost = sess.run([train_step, loss], feed_dict={
                        A_in: X_imgs[X_indices_mini_batch[0]],
                        P_in: X_imgs[X_indices_mini_batch[1]],
                        N_in: X_imgs[X_indices_mini_batch[2]],
                        is_training: True,
                        keep_prob: 0.5
                        })
                    print("{} mini-batch > {}/{} cost: {} dist-cost: {}".format(
                        epoch, batch_index, train_size // mini_batch_size, temp_cost / mini_batch_size, temp_dist_cost / mini_batch_size), end="\r")
                    temp_cost /= train_size
                    train_cost += temp_cost

            # test
            dev_batch_size = dev_size // 1000 # 随机选取 0.1% 训练集数据用来训练
            dev_permutation = list(np.random.permutation(dev_batch_size))
            dev_cost  = sess.run(loss, feed_dict={
                A_in: X_imgs[X_indices_dev_set[0][dev_permutation]],
                P_in: X_imgs[X_indices_dev_set[1][dev_permutation]],
                N_in: X_imgs[X_indices_dev_set[2][dev_permutation]],
                is_training: False,
                keep_prob: 1
                })
            dev_cost /= dev_batch_size
            print("{}/{} {} train cost is {} , dev cost is {}".format(
                epoch, num_epochs, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), train_cost, dev_cost))
        save(sess)
