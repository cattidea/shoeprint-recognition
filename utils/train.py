import os
import time
import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.data import data_import, gen_mini_batch, test_data_import
from utils.nn import model, triplet_loss, random_mini_batches, save, compute_embeddings
from utils.imager import H as IH, W as IW, plot
from utils.test import init_test_graph, data_test


MARGIN = 0.2


def init_graph():
    """ 初始化 IO 变量 """
    A = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="A")
    P = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="P")
    N = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="N")
    is_training = tf.placeholder(dtype=tf.bool, name="is_training")
    keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    A_emb = model(A, is_training, keep_prob)
    P_emb = model(P, is_training, keep_prob)
    N_emb = model(N, is_training, keep_prob)
    ops = {
        "A": A,
        "P": P,
        "N": N,
        "A_emb": A_emb,
        "P_emb": P_emb,
        "N_emb": N_emb,
        "is_training": is_training,
        "keep_prob": keep_prob
    }
    return ops

def train():
    """ 训练 """
    learning_rate = 0.0001
    num_epochs = 5000
    GPU = True
    class_per_batch = 8
    shoe_per_class = 8
    img_per_shoe = 6
    step = 512 # 计算 embeddings 时所用的步长
    test_step = 50

    # train data
    data_set = data_import(amplify=img_per_shoe)
    X_imgs = data_set["X_imgs"]
    indices = data_set["indices"]
    train_size = len(indices)

    # test data
    train_test_img_arrays, train_test_data_map = test_data_import(set_type="train")
    dev_test_img_arrays, dev_test_data_map = test_data_import(set_type="dev")
    train_scope_length = len(train_test_data_map[list(train_test_data_map.keys())[0]]["scope_indices"])
    dev_scope_length = len(dev_test_data_map[list(dev_test_data_map.keys())[0]]["scope_indices"])


    # GPU Config
    config = tf.ConfigProto(allow_soft_placement=True)
    if GPU:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # train 计算图
    ops = init_graph()
    print(ops["A"].name, ops["A_emb"].name, ops["is_training"].name, ops["keep_prob"].name)
    print(ops["P"].name, ops["P_emb"].name, ops["is_training"].name, ops["keep_prob"].name)
    print(ops["N"].name, ops["N_emb"].name, ops["is_training"].name, ops["keep_prob"].name)
    loss = triplet_loss(ops["A_emb"], ops["P_emb"], ops["N_emb"], MARGIN)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)
    embeddings_ops = {
        "input": ops["A"],
        "embeddings": ops["A_emb"],
        "is_training": ops["is_training"],
        "keep_prob": ops["keep_prob"]
    }

    # test 计算图
    train_test_embeddings_shape = (len(train_test_img_arrays), *embeddings_ops["embeddings"].shape[1: ])
    dev_test_embeddings_shape = (len(dev_test_img_arrays), *embeddings_ops["embeddings"].shape[1: ])
    train_test_ops = init_test_graph(train_scope_length, train_test_embeddings_shape)
    dev_test_ops = init_test_graph(dev_scope_length, dev_test_embeddings_shape)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        clock = time.time()
        for epoch in range(1, num_epochs+1):
            # train
            train_costs = []
            for batch_index, triplets in gen_mini_batch(
                indices, class_per_batch=class_per_batch, shoe_per_class=shoe_per_class, img_per_shoe=img_per_shoe,
                img_arrays=X_imgs, sess=sess, ops=embeddings_ops, alpha=MARGIN, step=step):

                triplet_list = [list(line) for line in zip(*triplets)]
                if not triplet_list:
                    continue
                mini_batch_size = len(triplet_list[0])

                _, temp_cost = sess.run([train_step, loss], feed_dict={
                    ops["A"]: X_imgs[triplet_list[0]],
                    ops["P"]: X_imgs[triplet_list[1]],
                    ops["N"]: X_imgs[triplet_list[2]],
                    ops["is_training"]: True,
                    ops["keep_prob"]: 0.5
                    })
                print("{} mini-batch > {}/{} cost: {} ".format(
                    epoch, batch_index, train_size, temp_cost / mini_batch_size), end="\r")
                temp_cost /= mini_batch_size
                train_costs.append(temp_cost)
            train_cost = sum(train_costs) / len(train_costs)

            # test
            if epoch % test_step == 0:
                train_test_embeddings = compute_embeddings(train_test_img_arrays, sess=sess, ops=embeddings_ops, step=step)
                dev_test_embeddings = compute_embeddings(dev_test_img_arrays, sess=sess, ops=embeddings_ops, step=step)
                _, train_rate = data_test(train_test_data_map, train_test_embeddings, sess, train_test_ops, log=False)
                _, dev_rate = data_test(dev_test_data_map, dev_test_embeddings, sess, dev_test_ops, log=False)

                prec_time_stamp = (time.time() - clock) * ((num_epochs - epoch) // test_step) + clock
                clock = time.time()
                print("{}/{} {} train cost is {} train prec is {:.2%} dev prec is {:.2%} >> {} ".format(
                    epoch, num_epochs, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), train_cost,
                    train_rate, dev_rate, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(prec_time_stamp))))
            else:
                print("{}/{} {} train cost is {}".format(
                    epoch, num_epochs, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), train_cost))
        save(sess)
