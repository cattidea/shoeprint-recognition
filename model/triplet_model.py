import numpy as np
import tensorflow as tf

from config_parser.config import IH, IW, MARGIN, PATHS
from model.base import ModelBase


class TripletModel(ModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.ops = {}
        self.test_ops = {}
        self.debug_op = None

    def model(self, X, is_training, keep_prob):
        raise NotImplementedError

    def init_ops(self):
        """ 初始化训练计算图 """
        learning_rate_value = self.config["learning_rate"]

        A = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="A")
        P = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="P")
        N = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="N")
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        A_emb = self.model(A, is_training, keep_prob)
        P_emb = self.model(P, is_training, keep_prob)
        N_emb = self.model(N, is_training, keep_prob)
        loss = self.triplet_loss(A_emb, P_emb, N_emb, MARGIN)
        learning_rate = tf.Variable(
            learning_rate_value, trainable=False, name="learning_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(loss)
        tf.add_to_collection("emb", loss)
        tf.add_to_collection("emb", train_step)

        self.ops = {
            "A": A,
            "P": P,
            "N": N,
            "A_emb": A_emb,
            "P_emb": P_emb,
            "N_emb": N_emb,
            "is_training": is_training,
            "keep_prob": keep_prob,
            "loss": loss,
            "train_step": train_step,
            "learning_rate": learning_rate
        }

    def get_ops_from_graph(self, graph):
        """ 从已有模型中恢复计算图 """
        A = graph.get_tensor_by_name("A:0")
        P = graph.get_tensor_by_name("P:0")
        N = graph.get_tensor_by_name("N:0")
        A_emb = graph.get_tensor_by_name("l2_normalize:0")
        P_emb = graph.get_tensor_by_name("l2_normalize_1:0")
        N_emb = graph.get_tensor_by_name("l2_normalize_2:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        learning_rate = graph.get_tensor_by_name("learning_rate:0")
        loss, train_step = tf.get_collection("emb")

        self.ops = {
            "A": A,
            "P": P,
            "N": N,
            "A_emb": A_emb,
            "P_emb": P_emb,
            "N_emb": N_emb,
            "is_training": is_training,
            "keep_prob": keep_prob,
            "loss": loss,
            "train_step": train_step,
            "learning_rate": learning_rate
        }

    def init_test_ops(self, name, scope_length, num_augment, embeddings_length):
        """ 初始化测试计算图
        可根据 name 创建多个测试计算图
        """
        embeddings_shape = (embeddings_length, *self.ops["A_emb"].shape[1:])
        origin_indices = tf.placeholder(
            dtype=tf.int32, shape=(num_augment, ), name="origin_indices")
        scope_indices = tf.placeholder(dtype=tf.int32, shape=(
            scope_length, ), name="scope_indices")
        embeddings_op = tf.placeholder(
            dtype=tf.float32, shape=embeddings_shape, name="embeddings")

        origin_embeddings = tf.gather(embeddings_op, origin_indices)
        scope_embeddings = tf.gather(embeddings_op, scope_indices)
        scope_embeddings = tf.stack(
            [scope_embeddings for _ in range(num_augment)], axis=1)
        res_op = tf.reduce_min(tf.reduce_sum(tf.square(tf.subtract(
            origin_embeddings, scope_embeddings)), axis=-1), axis=-1)
        min_index_op = tf.argmin(res_op)
        self.test_ops[name] = {
            "origin_indices": origin_indices,
            "scope_indices": scope_indices,
            "embeddings": embeddings_op,
            "res": res_op,
            "min_index": min_index_op
        }

    def compute_embeddings(self, input, sess):
        """ 计算嵌入 """
        step = self.config["emb_step"]
        ops = {
            "input": self.ops["N"],
            "embeddings": self.ops["N_emb"],
            "is_training": self.ops["is_training"],
            "keep_prob": self.ops["keep_prob"]
        }

        array_length = len(input)
        embeddings = np.zeros(
            (array_length, *ops["embeddings"].shape[1:]), dtype=np.float32)

        for i in range(0, array_length, step):
            if array_length > 10 * step:
                print("compute embeddings {}/{} ".format(i, array_length), end="\r")
            input_batch = input[i: i + step]
            embeddings_batch = sess.run(ops["embeddings"], feed_dict={
                ops["input"]: np.divide(input_batch, 127.5, dtype=np.float32) - 1,
                ops["is_training"]: False,
                ops["keep_prob"]: 1
            })
            embeddings[i: i+step] = embeddings_batch
        return embeddings

    def update_learning_rate(self, sess):
        """ 更新 learning rate """
        new_learning_rate = self.config["learning_rate"]
        learning_rate_op = self.ops["learning_rate"]
        origin_learning_rate = sess.run(learning_rate_op)
        if abs(origin_learning_rate - new_learning_rate) / new_learning_rate > 1e-6:
            sess.run(tf.assign(learning_rate_op, new_learning_rate))
            print(
                "update learning rate {} -> {}".format(origin_learning_rate, new_learning_rate))

    @staticmethod
    def triplet_loss(anchor, positive, negative, alpha=0.2):
        """ 计算三元组损失 """
        pos_dist = tf.reduce_sum(
            tf.square(tf.subtract(anchor, positive)), axis=-1)
        neg_dist = tf.reduce_sum(
            tf.square(tf.subtract(anchor, negative)), axis=-1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
        return loss
