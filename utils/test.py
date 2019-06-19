import os
import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.data import test_data_import
from utils.nn import model, restore, compute_embeddings


CONFIG = Config()
RESULT_FILE = CONFIG['result_file']
MODEL_PATH = CONFIG['model_path']
MODEL_DIR = CONFIG['model_dir']
MODEL_META = CONFIG['model_meta']


def get_emb_ops(graph):
    A = graph.get_tensor_by_name("A:0")
    P = graph.get_tensor_by_name("P:0")
    N = graph.get_tensor_by_name("N:0")
    A_emb = graph.get_tensor_by_name("l2_normalize:0")
    P_emb = graph.get_tensor_by_name("l2_normalize_1:0")
    N_emb = graph.get_tensor_by_name("l2_normalize_2:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
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


def init_test_graph(scope_length, embeddings_shape):
    """ 初始化测试计算图 """
    origin_index = tf.placeholder(dtype=tf.int32, shape=(), name="origin_index")
    scope_indices = tf.placeholder(dtype=tf.int32, shape=(scope_length), name="scope_indices")
    embeddings_op = tf.placeholder(dtype=tf.float32, shape=embeddings_shape, name="scope_indices")

    origin_embeddings = tf.gather(embeddings_op, [origin_index for _ in range(scope_length)])
    scope_embeddings = tf.gather(embeddings_op, scope_indices)
    res_op = tf.reduce_sum(tf.square(tf.subtract(origin_embeddings, scope_embeddings)), axis=-1)
    min_index_op = tf.argmin(res_op)
    ops = {
        "origin_index": origin_index,
        "scope_indices": scope_indices,
        "embeddings": embeddings_op,
        "res": res_op,
        "min_index": min_index_op
    }
    return ops


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
        with tf.Session(graph=graph) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

            ops = get_emb_ops(graph)

            embedding_ops = {
                "input": ops["A"],
                "embeddings": ops["A_emb"],
                "is_training": ops["is_training"],
                "keep_prob": ops["keep_prob"]
            }
            embeddings = compute_embeddings(img_arrays, sess=sess, ops=embedding_ops, step=step)

            # 测试计算图
            embeddings_shape = (len(img_arrays), *embedding_ops["embeddings"].shape[1: ])
            test_ops = init_test_graph(scope_length, embeddings_shape)

            res, rate = data_test(test_data_map, embeddings, sess, test_ops, log=True)
            # print(results)

