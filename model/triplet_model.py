import numpy as np
import tensorflow as tf

from config_parser.config import IH, IW, MARGIN, PATHS, k_H, k_W
from model.base import ModelBase


class TripletModel(ModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.ops = {}
        self.test_ops = {}
        self.debug_op = None


    def model(self, X, mask, is_training, keep_prob):
        raise NotImplementedError


    def init_ops(self):
        """ 初始化训练计算图 """
        learning_rate = self.config["learning_rate"]
        A = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="A")
        P = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="P")
        N = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="N")
        A_masks = tf.placeholder(dtype=tf.float32, shape=[None, IH//k_H, IW//k_W, 1], name="A_masks")
        P_masks = tf.placeholder(dtype=tf.float32, shape=[None, IH//k_H, IW//k_W, 1], name="P_masks")
        N_masks = tf.placeholder(dtype=tf.float32, shape=[None, IH//k_H, IW//k_W, 1], name="N_masks")
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        A_emb = self.model(A, A_masks, is_training, keep_prob)
        P_emb = self.model(P, P_masks, is_training, keep_prob)
        N_emb = self.model(N, N_masks, is_training, keep_prob)
        loss = self.triplet_loss(A_emb, P_emb, N_emb, MARGIN)
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
            "A_masks": A_masks,
            "P_masks": P_masks,
            "N_masks": N_masks,
            "A_emb": A_emb,
            "P_emb": P_emb,
            "N_emb": N_emb,
            "is_training": is_training,
            "keep_prob": keep_prob,
            "loss": loss,
            "train_step": train_step
        }

    def get_ops_from_graph(self, graph):
        """ 从已有模型中恢复计算图 """
        A = graph.get_tensor_by_name("A:0")
        P = graph.get_tensor_by_name("P:0")
        N = graph.get_tensor_by_name("N:0")
        A_masks = graph.get_tensor_by_name("A_masks:0")
        P_masks = graph.get_tensor_by_name("P_masks:0")
        N_masks = graph.get_tensor_by_name("N_masks:0")
        A_emb = graph.get_tensor_by_name("l2_normalize:0")
        P_emb = graph.get_tensor_by_name("l2_normalize_1:0")
        N_emb = graph.get_tensor_by_name("l2_normalize_2:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        loss, train_step = tf.get_collection("emb")
        self.ops = {
            "A": A,
            "P": P,
            "N": N,
            "A_masks": A_masks,
            "P_masks": P_masks,
            "N_masks": N_masks,
            "A_emb": A_emb,
            "P_emb": P_emb,
            "N_emb": N_emb,
            "is_training": is_training,
            "keep_prob": keep_prob,
            "loss": loss,
            "train_step": train_step
        }


    def init_test_ops(self, name, scope_length, num_amplify, embeddings_length):
        """ 初始化测试计算图
        可根据 name 创建多个测试计算图
        """
        embeddings_shape = (embeddings_length, *self.ops["A_emb"].shape[1: ])
        origin_indices = tf.placeholder(dtype=tf.int32, shape=(num_amplify, ), name="origin_indices")
        scope_indices = tf.placeholder(dtype=tf.int32, shape=(scope_length, ), name="scope_indices")
        embeddings_op = tf.placeholder(dtype=tf.float32, shape=embeddings_shape, name="embeddings")

        origin_embeddings = tf.gather(embeddings_op, origin_indices)
        scope_embeddings = tf.gather(embeddings_op, scope_indices)
        scope_embeddings = tf.stack([scope_embeddings for _ in range(num_amplify)], axis=1)
        res_op = tf.reduce_min(tf.reduce_sum(tf.square(tf.subtract(origin_embeddings, scope_embeddings)), axis=-1), axis=-1)
        min_index_op = tf.argmin(res_op)
        self.test_ops[name] = {
            "origin_indices": origin_indices,
            "scope_indices": scope_indices,
            "embeddings": embeddings_op,
            "res": res_op,
            "min_index": min_index_op
        }


    def compute_embeddings(self, input, masks, sess):
        """ 计算嵌入 """
        step = self.config["emb_step"]
        ops = {
            "input": self.ops["N"],
            "masks": self.ops["N_masks"],
            "embeddings": self.ops["N_emb"],
            "is_training": self.ops["is_training"],
            "keep_prob": self.ops["keep_prob"]
        }

        array_length = len(input)
        embeddings = np.zeros((array_length, *ops["embeddings"].shape[1: ]), dtype=np.float32)

        for i in range(0, array_length, step):
            if array_length > 10 * step:
                print("compute embeddings {}/{} ".format(i, array_length), end="\r")
            input_batch = input[i: i + step]
            masks_batch = masks[i: i + step]
            embeddings_batch = sess.run(ops["embeddings"], feed_dict={
                ops["input"]: np.divide(input_batch, 255, dtype=np.float32),
                ops["masks"]: masks_batch.astype(np.float32),
                ops["is_training"]: False,
                ops["keep_prob"]: 1
                })
            embeddings[i: i+step] = embeddings_batch
        return embeddings


    @staticmethod
    def triplet_loss(anchor, positive, negative, alpha = 0.2):
        """ 计算三元组损失 """
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
        return loss


    @staticmethod
    def maxout(inputs, num_units, axis=None):
        """ 将前层部分参数作为 maxout 的参数进行处理 """
        shape = inputs.get_shape().as_list()
        if axis is None:
            # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError('number of features({}) is not a multiple of num_units({})'
                .format(num_channels, num_units))
        shape[axis] = num_units
        shape += [num_channels // num_units]
        for i in range(len(shape)):
            if shape[i] is None:
                shape[i] = -1
        outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keepdims=False)
        return outputs


    def maxout_activation(self, num_units):
        return lambda x: self.maxout(x, num_units=num_units)


    @staticmethod
    def conv2d(X, scope, filter, kernel_size=3, strides=1, padding="same", activation=None, batch_norm=False, is_training=False):
        """ 卷积，带 BN 层 """
        X = tf.layers.conv2d(inputs=X, filters=filter, kernel_size=kernel_size, strides=strides, padding=padding,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_CONV", reuse=tf.AUTO_REUSE)
        if batch_norm:
            X = tf.layers.batch_normalization(inputs=X, training=is_training, name=scope+"_BN", reuse=tf.AUTO_REUSE)
        if activation:
            X = activation(X)
        return X


    @staticmethod
    def dense(X, name, units, activation=None, keep_prob=1):
        X = tf.layers.dense(inputs=X, units=units, name=name, reuse=tf.AUTO_REUSE)
        if keep_prob != 1:
            X = tf.nn.dropout(X, keep_prob)
        if activation:
            X = activation(X)
        return X


    @staticmethod
    def inception_v2(input, filters, scope, batch_norm=False, activation=None, is_training=False):
        """ inception_v2 网络
        ref: https://blog.csdn.net/loveliuzz/article/details/79135583
        """
        assert filters % 32 == 0
        k = filters // 32
        res_1_1 = tf.layers.conv2d(inputs=input, filters=8*k, kernel_size=1, strides=1, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_1_1", reuse=tf.AUTO_REUSE)
        res_1_1_t3 = tf.layers.conv2d(inputs=input, filters=12*k, kernel_size=1, strides=1, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_1_1_t3", reuse=tf.AUTO_REUSE)
        res_3_3 = tf.layers.conv2d(inputs=res_1_1_t3, filters=16*k, kernel_size=3, strides=1, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_3_3", reuse=tf.AUTO_REUSE)
        res_1_1_t5 = tf.layers.conv2d(inputs=input, filters=k, kernel_size=1, strides=1, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_1_1_t5", reuse=tf.AUTO_REUSE)
        res_3_3_t5 = tf.layers.conv2d(inputs=res_1_1_t5, filters=2*k, kernel_size=3, strides=1, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_3_3_t5", reuse=tf.AUTO_REUSE)
        res_5_5 = tf.layers.conv2d(inputs=res_3_3_t5, filters=4*k, kernel_size=3, strides=1, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_5_5", reuse=tf.AUTO_REUSE)
        res_pool_t = tf.layers.max_pooling2d(input, pool_size=3, strides=1, padding='same')
        res_pool = tf.layers.conv2d(inputs=res_pool_t, filters=4*k, kernel_size=1, strides=1, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_pool", reuse=tf.AUTO_REUSE)
        res = tf.concat([res_1_1, res_3_3, res_5_5, res_pool], axis=-1)
        if batch_norm:
            res = tf.layers.batch_normalization(inputs=res, training=is_training, name=scope+"_BN", reuse=tf.AUTO_REUSE)
        if activation:
            res = activation(res)
        return res
