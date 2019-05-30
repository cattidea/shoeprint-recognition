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
    # Placeholder:0 Placeholder_1:0 Placeholder_2:0 Relu_4:0 Relu_9:0 Relu_14:0
    X = graph.get_tensor_by_name("Placeholder:0")
    Y = graph.get_tensor_by_name("Relu_4:0")

    results = []

    origin_encodings = sess.run(Y, feed_dict={X: test_data_set_origin})
    for i in range(imgs_num):
        scope = test_data_set_scope[i]
        label = labels[i]
        origin_encoding = tf.gather(origin_encodings, [i for _ in range(scope_length)])
        scope_encoding = sess.run(Y, feed_dict={X: scope})
        # res = dist(scope_encoding, origin_encoding)
        # print(scope_encoding)
        res = tf.reduce_sum(tf.square(tf.subtract(origin_encoding, scope_encoding)), axis=-1)
        print(sess.run(res))
        min_index = sess.run(tf.argmin(res))[0][0]
        print(i, min_index, label[1])
        print("right" if min_index == label[1] else "wrong")
        results.append(res)
    print(results)

