import os
import time

import numpy as np
import tensorflow as tf

from config_parser.config import PATHS, SEED, TRAIN_HYPER_PARAMS, GPU
from data_loader.batch_loader import BatchLoader
from data_loader.data_loader import data_import, test_data_import
from data_loader.image import ALL, TRANSPOSE
from infer.test import data_test
from model.models import Model
from trainer.recorder import Recorder

RECORDER_PATH = PATHS["recorder_path"]


def train():
    """ 训练 """
    resume = TRAIN_HYPER_PARAMS["resume"]
    num_epochs = TRAIN_HYPER_PARAMS["num_epochs"]
    keep_prob = TRAIN_HYPER_PARAMS["keep_prob"]
    class_per_batch = TRAIN_HYPER_PARAMS["class_per_batch"]
    shoe_per_class = TRAIN_HYPER_PARAMS["shoe_per_class"]
    img_per_shoe = TRAIN_HYPER_PARAMS["img_per_shoe"]
    save_step = TRAIN_HYPER_PARAMS["save_step"]
    test_step = TRAIN_HYPER_PARAMS["test_step"]
    train_test = TRAIN_HYPER_PARAMS["train_test"]
    dev_test = TRAIN_HYPER_PARAMS["dev_test"]
    max_mini_batch_size = class_per_batch * \
        shoe_per_class * (shoe_per_class-1) / 2

    # GPU Config
    config = tf.ConfigProto()
    if GPU.enable:
        config.gpu_options.per_process_gpu_memory_fraction = GPU.memory_fraction
        config.gpu_options.allow_growth = True
        os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(map(lambda x: str(x), GPU.devices))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model = Model(TRAIN_HYPER_PARAMS)
    recorder = Recorder(RECORDER_PATH, resume=resume)
    recorder.upload_params(TRAIN_HYPER_PARAMS)

    # train data
    data_set = data_import(augment=ALL)
    img_arrays = data_set["img_arrays"]
    indices = data_set["indices"]
    train_size = len(indices)

    # test data
    if train_test or dev_test:
        test_img_arrays, test_data_map, _ = test_data_import(
            augment=[TRANSPOSE], action_type="train")
        train_scope_length = len(test_data_map["train"][0]["scope_indices"])
        train_num_augment = len(test_data_map["train"][0]["indices"])
        dev_scope_length = len(test_data_map["dev"][0]["scope_indices"])
        dev_num_augment = len(test_data_map["dev"][0]["indices"])

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(SEED)
        if resume:
            model.import_meta_graph()
            model.get_ops_from_graph(graph)
        else:
            model.init_ops()

        # test 计算图
        if train_test or dev_test:
            test_embeddings_length = len(test_img_arrays)
        if train_test:
            model.init_test_ops("train", train_scope_length,
                                train_num_augment, test_embeddings_length)
        if dev_test:
            model.init_test_ops("dev", dev_scope_length,
                                dev_num_augment, test_embeddings_length)

        with tf.Session(graph=graph, config=config) as sess:
            if resume:
                model.load(sess)
                print("成功恢复模型 {}".format(model.name))
            else:
                model.init_saver()
                sess.run(tf.global_variables_initializer())

            model.update_learning_rate(sess)

            clock = time.time()
            for epoch in range(recorder.checkpoint+1, num_epochs+1):
                recorder.update_checkpoint(epoch)
                # train
                train_costs = []
                triplet_cache = []
                for batch_index, triplets in BatchLoader(
                        model, indices, class_per_batch=class_per_batch, shoe_per_class=shoe_per_class, img_per_shoe=img_per_shoe,
                        img_arrays=img_arrays, sess=sess):

                    # 小数据 cache 机制
                    if len(triplets) == 0:
                        continue
                    elif len(triplets) + len(triplet_cache) <= max_mini_batch_size // 2:
                        triplet_cache.extend(triplets)
                        continue
                    elif max_mini_batch_size // 2 < len(triplets) + len(triplet_cache) <= max_mini_batch_size:
                        triplets.extend(triplet_cache)
                        triplet_cache.clear()

                    triplet_list = [list(line) for line in zip(*triplets)]
                    mini_batch_size = len(triplet_list[0])

                    _, temp_cost = sess.run([model.ops["train_step"], model.ops["loss"]], feed_dict={
                        model.ops["A"]: np.divide(img_arrays[triplet_list[0]], 127.5, dtype=np.float32) - 1,
                        model.ops["P"]: np.divide(img_arrays[triplet_list[1]], 127.5, dtype=np.float32) - 1,
                        model.ops["N"]: np.divide(img_arrays[triplet_list[2]], 127.5, dtype=np.float32) - 1,
                        model.ops["is_training"]: True,
                        model.ops["keep_prob"]: keep_prob
                    })
                    temp_cost /= max_mini_batch_size
                    print("{} mini-batch > {}/{} size: {} cost: {} ".format(
                        epoch, batch_index, train_size, mini_batch_size, temp_cost), end="\r")
                    train_costs.append(temp_cost)
                train_cost = sum(train_costs) / len(train_costs)

                # test
                log_str = "{}/{} {} train cost is {}".format(
                    epoch, num_epochs, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), train_cost)

                if epoch % test_step == 0:
                    train_top_1_acc, train_top_5_acc, dev_top_1_acc, dev_top_5_acc = "", "", "", ""
                    if train_test or dev_test:
                        test_embeddings = model.compute_embeddings(
                            test_img_arrays, sess=sess)

                    if train_test:
                        _, train_top_1_acc, train_top_5_acc = data_test(
                            test_data_map, "train", test_embeddings, sess, model, log=False)
                        log_str += " train top-1:{:.2%} top-5:{:.2%}" .format(train_top_1_acc, train_top_5_acc)
                    if dev_test:
                        _, dev_top_1_acc, dev_top_5_acc = data_test(
                            test_data_map, "dev", test_embeddings, sess, model, log=False)
                        log_str += " dev top-1:{:.2%} top-5:{:.2%}" .format(dev_top_1_acc, dev_top_5_acc)

                    # 预计完成时间
                    prec_time_stamp = (time.time() - clock) * \
                        ((num_epochs - epoch) // test_step) + clock
                    clock = time.time()
                    log_str += " >> {} ".format(time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(prec_time_stamp)))
                    recorder.record_item(epoch, [train_top_1_acc, train_top_5_acc, dev_top_1_acc, dev_top_5_acc])

                if epoch % save_step == 0:
                    model.save(sess)
                    recorder.save()

                print(log_str)
