import tensorflow as tf


from model.triplet_model import TripletModel


class ModelV1(TripletModel):

    def __init__(self, config):
        super().__init__(config)


    def model(self, X, mask, is_training, keep_prob):

        # CONV L1
        X = self.conv2d(X, scope="CONV_1", filter=64, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=3)

        # INCEPTION L1
        X = self.inception_v2(input=X, scope="INCEPTION_1", filters=64, activation=tf.nn.relu, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=2, strides=2)

        # INCEPTION L2
        X = self.inception_v2(input=X, scope="INCEPTION_2a", filters=128, activation=tf.nn.relu, is_training=is_training)
        X = self.inception_v2(input=X, scope="INCEPTION_2b", filters=128, activation=tf.nn.relu, is_training=is_training)
        X = self.inception_v2(input=X, scope="INCEPTION_2c", filters=128, activation=tf.nn.relu, is_training=is_training)

        # CONV to 1
        X = X * mask
        X = self.conv2d(X, scope="CONV_2a", filter=256, kernel_size=(7, 5), strides=3, padding="valid", activation=None, is_training=is_training)
        X = self.conv2d(X, scope="CONV_2b", filter=512, kernel_size=(6, 2), strides=1, padding="valid", activation=None, is_training=is_training)
        X = self.maxout(X, num_units=128)

        # flatten
        X = tf.layers.flatten(X)

        X = tf.nn.l2_normalize(X, axis=-1)
        return X
