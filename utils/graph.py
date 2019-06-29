import numpy as np
import tensorflow as tf


from utils.nn import model, triplet_loss
from utils.imager import H as IH, W as IW, plot


MARGIN = 0.2


def init_emb_ops(learning_rate=0.0001):
    """ 初始化训练计算图 """
    A = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="A")
    P = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="P")
    N = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, 1], name="N")
    is_training = tf.placeholder(dtype=tf.bool, name="is_training")
    keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    A_emb = model(A, is_training, keep_prob)
    P_emb = model(P, is_training, keep_prob)
    N_emb = model(N, is_training, keep_prob)
    loss = triplet_loss(A_emb, P_emb, N_emb, MARGIN)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)
    tf.add_to_collection("train", train_step)
    ops = {
        "A": A,
        "P": P,
        "N": N,
        "A_emb": A_emb,
        "P_emb": P_emb,
        "N_emb": N_emb,
        "is_training": is_training,
        "keep_prob": keep_prob,
        "loss": loss,
        "train_step": train_step
    }
    return ops


def get_emb_ops_from_graph(graph):
    """ 从已有模型中恢复计算图 """
    A = graph.get_tensor_by_name("A:0")
    P = graph.get_tensor_by_name("P:0")
    N = graph.get_tensor_by_name("N:0")
    A_emb = graph.get_tensor_by_name("l2_normalize:0")
    P_emb = graph.get_tensor_by_name("l2_normalize_1:0")
    N_emb = graph.get_tensor_by_name("l2_normalize_2:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    loss = graph.get_tensor_by_name("Sum_2:0")
    train_step = tf.get_collection("train")[0]
    ops = {
        "A": A,
        "P": P,
        "N": N,
        "A_emb": A_emb,
        "P_emb": P_emb,
        "N_emb": N_emb,
        "is_training": is_training,
        "keep_prob": keep_prob,
        "loss": loss,
        "train_step": train_step
    }
    return ops


def init_test_ops(scope_length, num_amplify, embeddings_shape):
    """ 初始化测试计算图 """
    origin_indices = tf.placeholder(dtype=tf.int32, shape=(num_amplify, ), name="origin_indices")
    scope_indices = tf.placeholder(dtype=tf.int32, shape=(scope_length, ), name="scope_indices")
    embeddings_op = tf.placeholder(dtype=tf.float32, shape=embeddings_shape, name="embeddings")

    origin_embeddings = tf.gather(embeddings_op, origin_indices)
    scope_embeddings = tf.gather(embeddings_op, scope_indices)
    scope_embeddings = tf.stack([scope_embeddings for _ in range(num_amplify)], axis=1)
    res_op = tf.reduce_min(tf.reduce_sum(tf.square(tf.subtract(origin_embeddings, scope_embeddings)), axis=-1), axis=-1)
    min_index_op = tf.argmin(res_op)
    ops = {
        "origin_indices": origin_indices,
        "scope_indices": scope_indices,
        "embeddings": embeddings_op,
        "res": res_op,
        "min_index": min_index_op
    }
    return ops
