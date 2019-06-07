import os
import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.data import test_data_import
from utils.nn import model, restore

DEBUG = True
CONFIG = Config()
RESULT_FILE = CONFIG['result_file']
DEBUG_FILE = CONFIG['debug_file']
MODEL_PATH = CONFIG['model_path']
MODEL_DIR = CONFIG['model_dir']
MODEL_META = CONFIG['model_meta']

def dist(A, B):
    return tf.reduce_sum(tf.square(tf.subtract(A, B)), axis=-1)

def get_variables():
    """ 初始化 IO 变量 """
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("A_in:0")
    Y = graph.get_tensor_by_name("A_out:0")
    return X, Y


def test():
    test_data_set_scope, test_data_set_origin, labels = test_data_import(DEBUG)

    imgs_num = len(test_data_set_scope)
    scope_length = len(test_data_set_scope[0])

    # sess = restore()
    # X, Y = get_variables()
    sess = tf.Session()
    saver = tf.train.import_meta_graph(MODEL_META)
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
    graph = tf.get_default_graph()

    A_in = graph.get_tensor_by_name("A_in:0")
    P_in = graph.get_tensor_by_name("P_in:0")
    N_in = graph.get_tensor_by_name("N_in:0")
    A_out = graph.get_tensor_by_name("Relu_5:0")
    P_out = graph.get_tensor_by_name("Relu_11:0")
    N_out = graph.get_tensor_by_name("Relu_17:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")

    X = A_in
    Y = A_out

    results = []

    origin_encodings = sess.run(Y, feed_dict={X: test_data_set_origin, is_training: False, keep_prob: 1})
    cnt = 0
    total = 0
    for i in range(imgs_num):
        scope = test_data_set_scope[i]
        label = labels[i]
        origin_encoding = tf.gather(origin_encodings, [i for _ in range(scope_length)])
        scope_encoding = sess.run(Y, feed_dict={X: scope, is_training: False, keep_prob: 1})
        res = tf.reduce_sum(tf.square(tf.subtract(origin_encoding, scope_encoding)), axis=-1)
        min_index = sess.run(tf.argmin(res))

        if min_index == label[1]:
            isRight = True
            cnt += 1
        else:
            isRight = False
        total += 1
        print("{} y^:{} y:{} res:{} {:.2%}".format(i, min_index, label[1], isRight, cnt/total))
        results.append(res)
    print("{:.2%}".format(cnt/total))
    # print(results)

