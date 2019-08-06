import os
import time
import numpy as np
import tensorflow as tf

from config_parser.config import PATHS, TRAIN_HYPER_PARAMS, TEST_PARAMS
from model.models import Model
from data_loader.data_loader import test_data_import
from data_loader.image import TRANSPOSE, ROTATE, BILATERAL_BLUR, img_plot
from data_loader.base import Cacher


RESULT_FILE = PATHS['result_file']
SIMPLE_EMB_CACHE = PATHS["simple_emb_cache"]
GLOBAL = {}


def data_test(test_data_map, set_type, embeddings, sess, model, log=False, plot=False):
    cnt = 0
    total = 0
    results = []
    test_ops = model.test_ops[set_type]
    for i, item in enumerate(test_data_map[set_type]):
        print("data test ({}) {}/{} ".format(set_type, i, len(test_data_map[set_type])), end="\r")
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

        if log:
            pred_dist = np.argsort(np.argsort(res))[label]
            print("{:3} y_pred:{:3} y_label:{:3} dist:{:2} {:.2%}".format(i, min_index, label, pred_dist, cnt/total))

        if plot:
            img_plot(
                GLOBAL["img_arrays"][indices[0]],
                GLOBAL["img_arrays"][item["scope_indices"][label]],
                GLOBAL["img_arrays"][item["scope_indices"][min_index]]
            )

        results.append(res)
    if log:
        print("{:.2%}".format(cnt/total))
    return results, cnt/total


def test(test_config):
    GPU = test_config["use_GPU"]
    use_cache = test_config["use_cache"]

    model = Model(TRAIN_HYPER_PARAMS)
    simple_cacher = Cacher(SIMPLE_EMB_CACHE)
    img_arrays, test_data_map, simple_length = test_data_import(amplify=[(TRANSPOSE, BILATERAL_BLUR)], action_type="test")
    GLOBAL["img_arrays"] = img_arrays

    scope_length = len(test_data_map["test"][0]["scope_indices"])
    num_amplify = len(test_data_map["test"][0]["indices"])

    config = tf.ConfigProto()
    if GPU:
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


    graph = tf.Graph()
    with graph.as_default():
        model.import_meta_graph()
        with tf.Session(graph=graph, config=config) as sess:
            model.load(sess)

            clock = time.time()
            model.get_ops_from_graph(graph)

            if use_cache and os.path.exists(SIMPLE_EMB_CACHE):
                simple_embs = simple_cacher.read()
                shoeprint_embs = model.compute_embeddings(img_arrays[simple_length: ], sess=sess)
                embeddings = np.concatenate((simple_embs, shoeprint_embs))
                print("成功读取预编码模板")
            else:
                embeddings = model.compute_embeddings(img_arrays, sess=sess)
                simple_embs = embeddings[: simple_length]
                simple_cacher.save(simple_embs)

            # 测试计算图
            embeddings_length = len(img_arrays)
            model.init_test_ops("test", scope_length, num_amplify, embeddings_length)

            res, rate = data_test(test_data_map, "test", embeddings, sess, model, log=True, plot=True)
            print(rate)
            print("{:.2f}s".format(time.time() - clock))

    with open(RESULT_FILE, "w") as f:
        for i, item in enumerate(res):
            f.write(test_data_map["test"][i]["name"])
            for dist in item:
                f.write(",")
                f.write(str(1 / dist))
            f.write("\n")
