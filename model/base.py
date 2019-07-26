import tensorflow as tf

from config_parser.config import  PATHS


MODEL_PATH = PATHS['model_path']
MODEL_DIR = PATHS['model_dir']
MODEL_META = PATHS['model_meta']


class ModelBase():
    """ 模型基类 """

    def __init__(self, config):
        self.config = config

    def import_meta_graph(self):
        """ 从模型中恢复计算图 """
        self.saver = tf.train.import_meta_graph(MODEL_META)

    def load(self, sess):
        """ 恢复模型参数 """
        self.saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

    def init_saver(self):
        """ 初始化 saver """
        self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])

    def save(self, sess):
        """ 保存参数 """
        self.saver.save(sess, MODEL_PATH)
