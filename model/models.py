import tensorflow as tf

from model.triplet_model import TripletModel


class ModelV1(TripletModel):

    def __init__(self, config):
        super().__init__(config)


    def model(self, X, mask, is_training, keep_prob):

        # CONV L1
        X = self.conv2d(X, scope="CONV_1", filter=8, kernel_size=3, strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=3)

        # CONV L2
        X = self.conv2d(X, scope="CONV_2", filter=16, kernel_size=3, strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=2, strides=2)

        # INCEPTION L2
        X = self.inception_v2(input=X, scope="INCEPTION_1a", filters=64, batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.inception_v2(input=X, scope="INCEPTION_1b", filters=128, batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.inception_v2(input=X, scope="INCEPTION_1c", filters=128, batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.inception_v2(input=X, scope="INCEPTION_1d", filters=128, batch_norm=True, activation=tf.nn.relu, is_training=is_training)

        # CONV to 1
        X = X * mask
        X = self.conv2d(X, scope="CONV_2a", filter=256, kernel_size=(7, 5), strides=3, padding="valid", batch_norm=True, activation=None, is_training=is_training)
        X = self.conv2d(X, scope="CONV_2b", filter=512, kernel_size=(6, 2), strides=1, padding="valid", batch_norm=True, activation=None, is_training=is_training)
        X = self.maxout(X, num_units=128)

        # flatten
        X = tf.layers.flatten(X)

        X = tf.nn.l2_normalize(X, axis=-1)
        return X


class ModelV2(TripletModel):

    def __init__(self, config):
        super().__init__(config)


    def model(self, X, mask, is_training, keep_prob):

        # CONV L1
        X = self.conv2d(X, scope="CONV_1", filter=64, kernel_size=3, strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=3)

        # INCEPTION L1
        X = self.inception_v2(input=X, scope="INCEPTION_1", filters=64, batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=2, strides=2)

        # INCEPTION L2
        X = X * mask
        X = self.inception_v2(input=X, scope="INCEPTION_2a", filters=128, batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = X * mask
        X = self.inception_v2(input=X, scope="INCEPTION_2b", filters=128, batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = X * mask
        X = self.inception_v2(input=X, scope="INCEPTION_2c", filters=128, batch_norm=True, activation=tf.nn.relu, is_training=is_training)

        # CONV to 1
        X = tf.layers.average_pooling2d(X, pool_size=(22, 8), strides=1, padding='valid')

        # flatten
        X = tf.layers.flatten(X)

        # FC1
        X = self.dense(X, name="FC1", units=512, activation=self.maxout_activation(128), keep_prob=keep_prob)

        # Output
        X = tf.nn.l2_normalize(X, axis=-1)
        return X


class Model(ModelV2):
    def __init__(self, config):
        super().__init__(config)
