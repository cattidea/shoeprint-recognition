import os
import time
import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.data import data_import, gen_mini_batch, test_data_import
from utils.nn import compute_embeddings
from utils.imager import plot, TRANSPOSE
from utils.test import data_test
from utils.graph import init_test_ops, init_emb_ops, get_emb_ops_from_graph


MARGIN = 0.2
CONFIG = Config()
MODEL_PATH = CONFIG['model_path']
MODEL_DIR = CONFIG['model_dir']
MODEL_META = CONFIG['model_meta']


def train(resume=False):
    """ 训练 """
    learning_rate = 0.0001
    num_epochs = 5000
    GPU = True
    class_per_batch = 8
    shoe_per_class = 8
    img_per_shoe = 6
    emb_step = 1024
    test_step = 50
    max_to_keep = 5
    train_test = False
    dev_test = True

    # train data
    data_set = data_import(amplify=img_per_shoe)
    img_arrays = data_set["img_arrays"]
    indices = data_set["indices"]
    masks = data_set["masks"]
    train_size = len(indices)

    # test data
    if train_test:
        train_test_img_arrays, train_test_masks, train_test_data_map = test_data_import(amplify=[TRANSPOSE], set_type="train")
        train_scope_length = len(train_test_data_map[0]["scope_indices"])
        train_num_amplify = len(train_test_data_map[0]["indices"])

    if dev_test:
        dev_test_img_arrays, dev_test_masks, dev_test_data_map = test_data_import(amplify=[TRANSPOSE], set_type="dev")
        dev_scope_length = len(dev_test_data_map[0]["scope_indices"])
        dev_num_amplify = len(dev_test_data_map[0]["indices"])

    # GPU Config
    config = tf.ConfigProto(allow_soft_placement=True)
    if GPU:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(-1)
        if resume:
            saver = tf.train.import_meta_graph(MODEL_META)
            ops = get_emb_ops_from_graph(graph)
        else:
            ops = init_emb_ops(learning_rate=learning_rate)

        print(ops["A"].name, ops["A_emb"].name, ops["is_training"].name, ops["keep_prob"].name)
        print(ops["P"].name, ops["P_emb"].name, ops["is_training"].name, ops["keep_prob"].name)
        print(ops["N"].name, ops["N_emb"].name, ops["is_training"].name, ops["keep_prob"].name)

        embeddings_ops = {
            "input": ops["A"],
            "masks": ops["A_masks"],
            "embeddings": ops["A_emb"],
            "is_training": ops["is_training"],
            "keep_prob": ops["keep_prob"]
        }

        # test 计算图
        if train_test:
            train_test_embeddings_shape = (len(train_test_img_arrays), *embeddings_ops["embeddings"].shape[1: ])
            train_test_ops = init_test_ops(train_scope_length, train_num_amplify, train_test_embeddings_shape)
        if dev_test:
            dev_test_embeddings_shape = (len(dev_test_img_arrays), *embeddings_ops["embeddings"].shape[1: ])
            dev_test_ops = init_test_ops(dev_scope_length, dev_num_amplify, dev_test_embeddings_shape)

        with tf.Session(graph=graph, config=config) as sess:
            if resume:
                saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
                print("成功恢复模型")
            else:
                saver = tf.train.Saver(max_to_keep=max_to_keep)
                sess.run(tf.global_variables_initializer())

            clock = time.time()
            for epoch in range(1, num_epochs+1):
                # train
                train_costs = []
                for batch_index, triplets in gen_mini_batch(
                    indices, class_per_batch=class_per_batch, shoe_per_class=shoe_per_class, img_per_shoe=img_per_shoe,
                    img_arrays=img_arrays, masks=masks, sess=sess, ops=embeddings_ops, alpha=MARGIN, step=emb_step):

                    triplet_list = [list(line) for line in zip(*triplets)]
                    if not triplet_list:
                        continue
                    mini_batch_size = len(triplet_list[0])

                    _, temp_cost = sess.run([ops["train_step"], ops["loss"]], feed_dict={
                        ops["A"]: (img_arrays[triplet_list[0]] / 255).astype(np.float32),
                        ops["P"]: (img_arrays[triplet_list[1]] / 255).astype(np.float32),
                        ops["N"]: (img_arrays[triplet_list[2]] / 255).astype(np.float32),
                        ops["A_masks"]: masks[triplet_list[0]].astype(np.float32),
                        ops["P_masks"]: masks[triplet_list[1]].astype(np.float32),
                        ops["N_masks"]: masks[triplet_list[2]].astype(np.float32),
                        ops["is_training"]: True,
                        ops["keep_prob"]: 0.5
                        })
                    print("{} mini-batch > {}/{} size: {} cost: {} ".format(
                        epoch, batch_index, train_size, mini_batch_size, temp_cost / mini_batch_size), end="\r")
                    temp_cost /= mini_batch_size
                    train_costs.append(temp_cost)
                train_cost = sum(train_costs) / len(train_costs)

                # test
                log_str = "{}/{} {} train cost is {}".format(
                    epoch, num_epochs, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), train_cost)
                if epoch % test_step == 0:
                    if train_test:
                        train_test_embeddings = compute_embeddings(train_test_img_arrays, train_test_masks, sess=sess, ops=embeddings_ops, step=emb_step)
                        _, train_rate = data_test(train_test_data_map, train_test_embeddings, sess, train_test_ops, log=False)
                        log_str += " train prec is {:.2%}" .format(train_rate)
                    if dev_test:
                        dev_test_embeddings = compute_embeddings(dev_test_img_arrays, dev_test_masks, sess=sess, ops=embeddings_ops, step=emb_step)
                        _, dev_rate = data_test(dev_test_data_map, dev_test_embeddings, sess, dev_test_ops, log=False)
                        log_str += " dev prec is {:.2%}" .format(dev_rate)

                    prec_time_stamp = (time.time() - clock) * ((num_epochs - epoch) // test_step) + clock
                    clock = time.time()
                    log_str += " >> {} ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(prec_time_stamp)))

                    saver.save(sess, MODEL_PATH)
                print(log_str)
