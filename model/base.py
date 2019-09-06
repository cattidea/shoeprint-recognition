import os

import tensorflow as tf

from config_parser.config import MODEL_DIR

MODEL_PATH = os.path.join(MODEL_DIR, "model.ckpt")
MODEL_META = os.path.join(MODEL_DIR, "model.ckpt.meta")


class ModelBase():
    """ 模型基类 """

    def __init__(self, config):
        self.config = config

    def import_meta_graph(self):
        """ 从模型中恢复计算图 """
        assert os.path.exists(MODEL_META), "模型文件夹无效"
        self.saver = tf.train.import_meta_graph(MODEL_META)

    def load(self, sess):
        """ 恢复模型参数 """
        assert os.path.exists(MODEL_META), "模型文件夹无效"
        self.saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

    def init_saver(self):
        """ 初始化 saver """
        self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])

    def save(self, sess):
        """ 保存参数 """
        self.saver.save(sess, MODEL_PATH)
