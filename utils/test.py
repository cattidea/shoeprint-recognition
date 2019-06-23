import os
import time
import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.data import test_data_import
from utils.nn import model, restore, compute_embeddings
from utils.imager import plot
from utils.graph import get_emb_ops_from_graph, init_test_ops


CONFIG = Config()
RESULT_FILE = CONFIG['result_file']
MODEL_PATH = CONFIG['model_path']
MODEL_DIR = CONFIG['model_dir']
MODEL_META = CONFIG['model_meta']


def data_test(test_data_map, embeddings, sess, test_ops, log=False):
    cnt = 0
    total = 0
    results = []
    for i, origin_name in enumerate(test_data_map):
        label = test_data_map[origin_name]["label"]
        index = test_data_map[origin_name]["index"]
        res, min_index = sess.run([test_ops["res"], test_ops["min_index"]], feed_dict={
            test_ops["origin_index"]: index,
            test_ops["scope_indices"]: test_data_map[origin_name]["scope_indices"],
            test_ops["embeddings"]: embeddings
            })

        isRight = min_index == label
        if isRight:
            cnt += 1
        total += 1
        if log:
            print("{:3} y_pred:{:3} y_label:{:3} res:{:2} {:.2%}".format(i, min_index, label, isRight, cnt/total))

        # plot(img_arrays[index])
        # plot(img_arrays[test_data_map[origin_name]["scope_indices"][label]])
        # plot(img_arrays[test_data_map[origin_name]["scope_indices"][min_index]])
        results.append(res)
    if log:
        print("{:.2%}".format(cnt/total))
    return results, cnt/total


def test():
    step = 512
    img_arrays, test_data_map = test_data_import(set_type="test")
    GPU = True

    imgs_num = len(list(test_data_map.keys()))
    scope_length = len(test_data_map[list(test_data_map.keys())[0]]["scope_indices"])
    array_num = len(img_arrays)

    config = tf.ConfigProto(allow_soft_placement=True)
    if GPU:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"


    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(MODEL_META)
        with tf.Session(graph=graph, config=config) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

            ops = get_emb_ops_from_graph(graph)

            embedding_ops = {
                "input": ops["A"],
                "embeddings": ops["A_emb"],
                "is_training": ops["is_training"],
                "keep_prob": ops["keep_prob"]
            }
            embeddings = compute_embeddings(img_arrays, sess=sess, ops=embedding_ops, step=step)

            # 测试计算图
            embeddings_shape = (len(img_arrays), *embedding_ops["embeddings"].shape[1: ])
            test_ops = init_test_ops(scope_length, embeddings_shape)

            clock = time.time()
            res, rate = data_test(test_data_map, embeddings, sess, test_ops, log=True)
            print("{:.2f}s".format(time.time() - clock))
            # print(results)

