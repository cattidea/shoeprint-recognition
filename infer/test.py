import os
import time

import numpy as np
import tensorflow as tf

from config_parser.config import PATHS, TEST_PARAMS, TRAIN_HYPER_PARAMS
from data_loader.base import Cacher
from data_loader.data_loader import test_data_import
from data_loader.image import BILATERAL_BLUR, ROTATE, TRANSPOSE, img_plot
from model.models import Model

RESULT_FILE = PATHS['result_file']
SIMPLE_EMB_CACHE = PATHS["simple_emb_cache"]
SEP = TEST_PARAMS["separator"]
IS_PLOT = TEST_PARAMS["plot"]
IS_LOG = TEST_PARAMS["log"]
GLOBAL = {}


def data_test(test_data_map, set_type, embeddings, sess, model, log=False, plot=False):
    """ 对构建好的数据进行测试 """
    cnt = 0
    total = 0
    results = []
    test_ops = model.test_ops[set_type]
    for i, item in enumerate(test_data_map[set_type]):
        print("data test ({}) {}/{} ".format(set_type,
                                             i, len(test_data_map[set_type])), end="\r")
        label = item["label"]
        indices = item["indices"]
        res, min_index = sess.run([test_ops["res"], test_ops["min_index"]], feed_dict={
            test_ops["origin_indices"]: indices,
            test_ops["scope_indices"]: item["scope_indices"],
            test_ops["embeddings"]: embeddings
        })

        # 计算预测值与真实标签是否相同
        isRight = min_index == label
        if isRight:
            cnt += 1
        total += 1

        if log:
            # 计算真实标签排在预测序列第几位（0 即为预测正确，数值越大偏差越大）
            pred_dist = np.argsort(np.argsort(res))[label]
            print("{:3} y_pred:{:3} y_label:{:3} dist:{:2} {:.2%}".format(
                i, min_index, label, pred_dist, cnt/total))

        if plot:
            img_plot(
                GLOBAL["img_arrays"][indices[0]],
                GLOBAL["img_arrays"][item["scope_indices"][label]],
                GLOBAL["img_arrays"][item["scope_indices"][min_index]]
            )

        results.append(res)
    if log:
        # 计算准确率大小
        print("{:.2%}".format(cnt/total))
    return results, cnt/total


def test(test_config):
    """ 测试主函数 """
    GPU = test_config["use_GPU"]
    use_cache = test_config["use_cache"]

    # 加载模型与数据
    model = Model(TRAIN_HYPER_PARAMS)
    simple_cacher = Cacher(SIMPLE_EMB_CACHE)
    img_arrays, test_data_map, simple_length = test_data_import(
        amplify=[(TRANSPOSE, BILATERAL_BLUR)], action_type="test")
    GLOBAL["img_arrays"] = img_arrays

    scope_length = len(test_data_map["test"][0]["scope_indices"])
    num_amplify = len(test_data_map["test"][0]["indices"])

    config = tf.ConfigProto()
    if GPU:
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # 启动 TF 计算图与会话
    graph = tf.Graph()
    with graph.as_default():
        model.import_meta_graph()
        with tf.Session(graph=graph, config=config) as sess:
            model.load(sess)

            clock = time.time()
            model.get_ops_from_graph(graph)

            # 计算嵌入
            if use_cache and os.path.exists(SIMPLE_EMB_CACHE):
                # 如果已经有编码过的样本嵌入则直接读取
                simple_embs = simple_cacher.read()
                shoeprint_embs = model.compute_embeddings(
                    img_arrays[simple_length:], sess=sess)
                embeddings = np.concatenate((simple_embs, shoeprint_embs))
                print("成功读取预编码模板")
            else:
                embeddings = model.compute_embeddings(img_arrays, sess=sess)
                simple_embs = embeddings[: simple_length]
                simple_cacher.save(simple_embs)

            # 初始化测试计算图
            embeddings_length = len(img_arrays)
            model.init_test_ops("test", scope_length,
                                num_amplify, embeddings_length)

            # 测试数据
            res, acc = data_test(
                test_data_map, "test", embeddings, sess, model, log=IS_LOG, plot=IS_PLOT)
            print(acc)
            print("{:.2f}s".format(time.time() - clock))

    # 将结果写入输出文件
    with open(RESULT_FILE, "w") as f:
        for i, item in enumerate(res):
            f.write(test_data_map["test"][i]["name"])
            for dist in item:
                f.write(SEP)
                f.write(str(1 / dist))
            f.write("\n")
