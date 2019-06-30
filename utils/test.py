import os
import time
import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.data import test_data_import
from utils.nn import model, compute_embeddings
from utils.imager import plot, TRANSPOSE, ROTATE
from utils.graph import get_emb_ops_from_graph, init_test_ops


CONFIG = Config()
RESULT_FILE = CONFIG['result_file']
MODEL_PATH = CONFIG['model_path']
MODEL_DIR = CONFIG['model_dir']
MODEL_META = CONFIG['model_meta']

GLOBAL = {}


def data_test(test_data_map, embeddings, sess, test_ops, log=False):
    cnt = 0
    total = 0
    results = []
    for i, item in enumerate(test_data_map):
        label = item["label"]
        indices = item["indices"]
        res, min_index = sess.run([test_ops["res"], test_ops["min_index"]], feed_dict={
            test_ops["origin_indices"]: indices,
            test_ops["scope_indices"]: item["scope_indices"],
            test_ops["embeddings"]: embeddings
            })

        isRight = min_index == label
        if isRight:
            cnt += 1
        total += 1

        # plot(GLOBAL["img_arrays"][indices[0]])
        # plot(GLOBAL["img_arrays"][item["scope_indices"][label]])
        # plot(GLOBAL["img_arrays"][item["scope_indices"][min_index]])

        if log:
            pred_dist = np.argsort(np.argsort(res))[label]
            print("{:3} y_pred:{:3} y_label:{:3} dist:{:2} {:.2%}".format(i, min_index, label, pred_dist, cnt/total))

        results.append(res)
    if log:
        print("{:.2%}".format(cnt/total))
    return results, cnt/total


def test():
    emb_step = 512
    img_arrays, masks, test_data_map = test_data_import(amplify=[TRANSPOSE], set_type="test")
    GLOBAL["img_arrays"] = img_arrays
    GPU = False

    scope_length = len(test_data_map[0]["scope_indices"])
    num_amplify = len(test_data_map[0]["indices"])

    config = tf.ConfigProto(allow_soft_placement=True)
    if GPU:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(MODEL_META)
        with tf.Session(graph=graph, config=config) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

            clock = time.time()
            ops = get_emb_ops_from_graph(graph)

            embedding_ops = {
                "input": ops["A"],
                "masks": ops["A_masks"],
                "embeddings": ops["A_emb"],
                "is_training": ops["is_training"],
                "keep_prob": ops["keep_prob"]
            }
            embeddings = compute_embeddings(img_arrays, masks, sess=sess, ops=embedding_ops, step=emb_step)

            # 测试计算图
            embeddings_shape = (len(img_arrays), *embedding_ops["embeddings"].shape[1: ])
            test_ops = init_test_ops(scope_length, num_amplify, embeddings_shape)

            res, rate = data_test(test_data_map, embeddings, sess, test_ops, log=True)
            print("{:.2f}s".format(time.time() - clock))

    with open(RESULT_FILE, "w") as f:
        for i, item in enumerate(res):
            f.write(test_data_map[i]["name"])
            for dist in item:
                f.write(",")
                f.write(str(1 / dist))
            f.write("\n")
