import os
import time
import numpy as np
import tensorflow as tf


from config_parser.config import TRAIN_HYPER_PARAMS, SEED, PATHS
from infer.test import data_test
from model.models import Model
from data_loader.image import TRANSPOSE, ALL
from data_loader.data_loader import data_import, test_data_import
from data_loader.batch_loader import BatchAll
from trainer.recorder import Recorder


RECORDER_PATH = PATHS["recorder_path"]


def train(train_config):
    """ 训练 """
    resume = train_config["resume"]
    GPU = train_config["use_GPU"]

    num_epochs = TRAIN_HYPER_PARAMS["num_epochs"]
    keep_prob = TRAIN_HYPER_PARAMS["keep_prob"]
    class_per_batch = TRAIN_HYPER_PARAMS["class_per_batch"]
    shoe_per_class = TRAIN_HYPER_PARAMS["shoe_per_class"]
    img_per_shoe = TRAIN_HYPER_PARAMS["img_per_shoe"]
    save_step = TRAIN_HYPER_PARAMS["save_step"]
    test_step = TRAIN_HYPER_PARAMS["test_step"]
    train_test = TRAIN_HYPER_PARAMS["train_test"]
    dev_test = TRAIN_HYPER_PARAMS["dev_test"]
    max_mini_batch_size = class_per_batch * shoe_per_class * (shoe_per_class-1) / 2

    # GPU Config
    config = tf.ConfigProto()
    if GPU:
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model = Model(TRAIN_HYPER_PARAMS)
    recorder = Recorder(RECORDER_PATH, resume=resume)
    recorder.upload_params(TRAIN_HYPER_PARAMS)

    # train data
    data_set = data_import(amplify=ALL)
    img_arrays = data_set["img_arrays"]
    indices = data_set["indices"]
    train_size = len(indices)

    # test data
    if train_test or dev_test:
        test_img_arrays, test_data_map, _ = test_data_import(amplify=[TRANSPOSE], action_type="train")
        train_scope_length = len(test_data_map["train"][0]["scope_indices"])
        train_num_amplify = len(test_data_map["train"][0]["indices"])
        dev_scope_length = len(test_data_map["dev"][0]["scope_indices"])
        dev_num_amplify = len(test_data_map["dev"][0]["indices"])

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
            model.init_test_ops("train", train_scope_length, train_num_amplify, test_embeddings_length)
        if dev_test:
            model.init_test_ops("dev", dev_scope_length, dev_num_amplify, test_embeddings_length)

        with tf.Session(graph=graph, config=config) as sess:
            if resume:
                model.load(sess)
                print("成功恢复模型")
            else:
                model.init_saver()
                sess.run(tf.global_variables_initializer())

            model.update_learning_rate(sess)

            clock = time.time()
            for epoch in range(recorder.checkpoint+1, num_epochs+1):
                recorder.update_checkpoint(epoch)
                # train
                train_costs = []
                for batch_index, triplets in BatchAll(
                    model, indices, class_per_batch=class_per_batch, shoe_per_class=shoe_per_class, img_per_shoe=img_per_shoe,
                    img_arrays=img_arrays, sess=sess):

                    triplet_list = [list(line) for line in zip(*triplets)]
                    if not triplet_list:
                        train_costs.append(0)
                        continue
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
                    train_rate, dev_rate = "", ""
                    if train_test or dev_test:
                        test_embeddings = model.compute_embeddings(test_img_arrays, sess=sess)

                    if train_test:
                        _, train_rate = data_test(test_data_map, "train", test_embeddings, sess, model, log=False)
                        log_str += " train prec is {:.2%}" .format(train_rate)
                    if dev_test:
                        _, dev_rate = data_test(test_data_map, "dev", test_embeddings, sess, model, log=False)
                        log_str += " dev prec is {:.2%}" .format(dev_rate)

                    prec_time_stamp = (time.time() - clock) * ((num_epochs - epoch) // test_step) + clock
                    clock = time.time()
                    log_str += " >> {} ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(prec_time_stamp)))
                    recorder.record_item(epoch, [train_rate, dev_rate])

                if epoch % save_step == 0:
                    model.save(sess)
                    recorder.save()

                print(log_str)
