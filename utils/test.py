import os
import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.data import test_data_import
from utils.nn import model, restore


CONFIG = Config()
RESULT_FILE = CONFIG['result_file']
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
    batch_size = 512
    img_arrays, test_data_map = test_data_import()
    GPU = True

    imgs_num = len(list(test_data_map.keys()))
    scope_length = len(test_data_map[list(test_data_map.keys())[0]]["scope_indices"])
    array_num = len(img_arrays)
    results = []

    config = tf.ConfigProto(allow_soft_placement=True)
    if GPU:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    # sess = restore()
    # X, Y = get_variables()

    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(MODEL_META)
        with tf.Session(graph=graph) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
            # 编码计算图
            A_in = graph.get_tensor_by_name("A_in:0")
            P_in = graph.get_tensor_by_name("P_in:0")
            N_in = graph.get_tensor_by_name("N_in:0")
            A_out = graph.get_tensor_by_name("l2_normalize:0")
            P_out = graph.get_tensor_by_name("l2_normalize_1:0")
            N_out = graph.get_tensor_by_name("l2_normalize_2:0")
            is_training = graph.get_tensor_by_name("is_training:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")

            X = A_in
            Y = A_out
            encodings = np.zeros((array_num, *Y.shape[1: ]), dtype=np.float32)

            for i in range(0, array_num, batch_size):
                img_arrays_batch = img_arrays[i: i + batch_size]
                encodings_batch = sess.run(Y, feed_dict={X: img_arrays_batch, is_training: False, keep_prob: 1})
                encodings[i: i+batch_size] = encodings_batch

            # 测试计算图
            origin_index = tf.placeholder(dtype=tf.int32, shape=(), name="origin_index")
            scope_indices = tf.placeholder(dtype=tf.int32, shape=(scope_length), name="scope_indices")

            origin_encodings = tf.gather(encodings, [origin_index for _ in range(scope_length)])
            scope_encodings = tf.gather(encodings, scope_indices)
            res_op = tf.reduce_sum(tf.square(tf.subtract(origin_encodings, scope_encodings)), axis=-1)
            min_index_op = tf.argmin(res_op)

            cnt = 0
            total = 0
            for i, origin_name in enumerate(test_data_map):
                label = test_data_map[origin_name]["label"]
                index = test_data_map[origin_name]["index"]
                res, min_index = sess.run([res_op, min_index_op], feed_dict={
                    origin_index: index,
                    scope_indices: test_data_map[origin_name]["scope_indices"]
                    })

                isRight = min_index == label
                if isRight:
                    cnt += 1
                total += 1
                print("{:3} y_pred:{:3} y_label:{:3} res:{:2} {:.2%}".format(i, min_index, label, isRight, cnt/total))
                results.append(res)
            print("{:.2%}".format(cnt/total))
            # print(results)

