import tensorflow as tf

from model.layers import (conv2d, deepwise_conv2d, dense_block, inception_v2,
                          maxout, anonymous_net_block, separable_conv2d, transition_layer)
from model.triplet_model import TripletModel


class ModelV1(TripletModel):

    def __init__(self, config):
        super().__init__(config)
        self.name = "inception"
        self.input_size = (129, 49)

    def model(self, X, is_training, keep_prob):
        # mask = (X + 1) / 2

        # CONV L1
        X = conv2d(X, scope="CONV_1", filter=32, kernel_size=3, strides=1,
                   padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        # mask = tf.layers.average_pooling2d(mask, pool_size=3, strides=2, padding="same")
        # mask = tf.where(mask>0.1, mask, tf.zeros_like(mask))
        # X = tf.cast(tf.greater(mask, 0), tf.float32) * X

        # INCEPTION L2
        Res = conv2d(X, scope="CONV_res_2", filter=64,
                     kernel_size=1, strides=2, padding="same")
        X = inception_v2(input=X, scope="INCEPTION_2a", filters=64,
                         batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = inception_v2(input=X, scope="INCEPTION_2b", filters=64,
                         batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        # mask = tf.layers.max_pooling2d(mask, pool_size=3, strides=2, padding="same")
        # X = tf.cast(tf.greater(mask, 0), tf.float32) * X
        X = X + Res

        # INCEPTION L3
        Res = conv2d(X, scope="CONV_res_3", filter=128,
                     kernel_size=1, strides=2, padding="same")
        X = tf.nn.relu(X)
        X = inception_v2(input=X, scope="INCEPTION_3a", filters=128,
                         batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = inception_v2(input=X, scope="INCEPTION_3b", filters=128,
                         batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        # mask = tf.layers.max_pooling2d(mask, pool_size=3, strides=2, padding="same")
        # X = tf.cast(tf.greater(mask, 0), tf.float32) * X
        X = X + Res

        # INCEPTION L4
        Res = conv2d(X, scope="CONV_res_4", filter=256,
                     kernel_size=1, strides=2, padding="same")
        X = tf.nn.relu(X)
        X = inception_v2(input=X, scope="INCEPTION_4a", filters=256,
                         batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = inception_v2(input=X, scope="INCEPTION_4b", filters=256,
                         batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = inception_v2(input=X, scope="INCEPTION_4c", filters=256,
                         batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res
        X = maxout(X, num_units=128)

        # dw conv
        X = deepwise_conv2d(X, scope="dw_conv", kernel_size=(
            X.shape[1], X.shape[2]), strides=1, padding="valid")

        # flatten
        X = tf.layers.flatten(X)

        # Output
        X = tf.nn.l2_normalize(X, axis=-1)
        return X


class ModelV2(TripletModel):

    def __init__(self, config):
        super().__init__(config)
        self.name = "Xception"
        self.input_size = (129, 49)

    def model(self, X, is_training, keep_prob):

        # CONV L1
        X = conv2d(X, scope="CONV_1", filter=32, kernel_size=3, strides=1,
                   padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")

        # XCEPTION L2
        Res = conv2d(X, scope="CONV_res_2", filter=128,
                     kernel_size=1, strides=2, padding="same")
        X = separable_conv2d(X, scope="XCEPTION_2a", filter=64, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = separable_conv2d(X, scope="XCEPTION_2b", filter=128, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res

        # XCEPTION L3
        Res = X
        X = tf.nn.relu(X)
        X = separable_conv2d(X, scope="XCEPTION_3a", filter=128, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = separable_conv2d(X, scope="XCEPTION_3b", filter=128, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = separable_conv2d(X, scope="XCEPTION_3c", filter=128, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = X + Res

        # XCEPTION L4
        Res = conv2d(X, scope="CONV_res_4", filter=128,
                     kernel_size=1, strides=2, padding="same")
        X = tf.nn.relu(X)
        X = separable_conv2d(X, scope="XCEPTION_4a", filter=128, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = separable_conv2d(X, scope="XCEPTION_4b", filter=128, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res

        # XCEPTION L5
        Res = conv2d(X, scope="CONV_res_5", filter=256,
                     kernel_size=1, strides=2, padding="same")
        X = tf.nn.relu(X)
        X = separable_conv2d(X, scope="XCEPTION_5a", filter=256, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = separable_conv2d(X, scope="XCEPTION_5b", filter=256, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res
        X = maxout(X, num_units=128)

        # dw conv to 1x1
        X = deepwise_conv2d(X, scope="dw_conv_to_1_1", kernel_size=(
            X.shape[1], X.shape[2]), strides=1, padding="valid")

        # flatten
        X = tf.layers.flatten(X)

        # Output
        X = tf.nn.l2_normalize(X, axis=-1)
        return X


class ModelV3(TripletModel):

    def __init__(self, config):
        super().__init__(config)
        self.name = "AnonymousNet"
        self.input_size = (129, 49)

    def model(self, X, is_training, keep_prob):
        # mask = (X + 1) / 2

        # AnonymousNet L1
        X = anonymous_net_block(input=X, filters=16, scope="AnonymousNet_1", strides=2,
                                activate=True, batch_norm=True, is_training=is_training)
        # mask = tf.layers.average_pooling2d(mask, pool_size=3, strides=2, padding="same")
        # mask = tf.where(mask>0.1, mask, tf.zeros_like(mask))
        # X = tf.cast(tf.greater(mask, 0), tf.float32) * X

        # AnonymousNet L2
        X = anonymous_net_block(input=X, filters=16, scope="AnonymousNet_2a", strides=1,
                                activate=True, batch_norm=True, is_training=is_training)
        X = anonymous_net_block(input=X, filters=32, scope="AnonymousNet_2b", strides=2,
                                activate=True, batch_norm=True, is_training=is_training)
        # mask = tf.layers.max_pooling2d(mask, pool_size=3, strides=2, padding="same")
        # X = tf.cast(tf.greater(mask, 0), tf.float32) * X

        # AnonymousNet L3
        X = anonymous_net_block(input=X, filters=32, scope="AnonymousNet_3a", strides=1,
                                activate=True, batch_norm=True, is_training=is_training)
        X = anonymous_net_block(input=X, filters=64, scope="AnonymousNet_3b", strides=1,
                                activate=True, batch_norm=True, is_training=is_training)

        # AnonymousNet L4
        X = anonymous_net_block(input=X, filters=64, scope="AnonymousNet_4a", strides=1,
                                activate=True, batch_norm=True, is_training=is_training)
        X = anonymous_net_block(input=X, filters=128, scope="AnonymousNet_4b", strides=2,
                                activate=True, batch_norm=True, is_training=is_training)
        # mask = tf.layers.max_pooling2d(mask, pool_size=3, strides=2, padding="same")
        # X = tf.cast(tf.greater(mask, 0), tf.float32) * X

        # AnonymousNet L5
        X = anonymous_net_block(input=X, filters=128, scope="AnonymousNet_5a", strides=1,
                                activate=True, batch_norm=True, is_training=is_training)
        X = anonymous_net_block(input=X, filters=128, scope="AnonymousNet_5b", strides=2,
                                activate=True, batch_norm=True, is_training=is_training)
        # mask = tf.layers.max_pooling2d(mask, pool_size=3, strides=2, padding="same")
        # X = tf.cast(tf.greater(mask, 0), tf.float32) * X

        # dw conv to 1x1
        X = deepwise_conv2d(X, scope="dw_conv_to_1_1", kernel_size=(
            X.shape[1], X.shape[2]), strides=1, padding="valid")

        # flatten
        X = tf.layers.flatten(X)

        # Output
        X = tf.nn.l2_normalize(X, axis=-1)
        return X


class ModelV4(TripletModel):

    def __init__(self, config):
        super().__init__(config)
        self.name = "Xception_deeper"
        self.input_size = (257, 97)

    def model(self, X, is_training, keep_prob):

        # CONV L1
        X = conv2d(X, scope="CONV_1", filter=16, kernel_size=3, strides=1,
                   padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")

        # XCEPTION L2
        Res = conv2d(X, scope="CONV_res_2", filter=128,
                     kernel_size=1, strides=2, padding="same")
        X = separable_conv2d(X, scope="XCEPTION_2a", filter=64, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = separable_conv2d(X, scope="XCEPTION_2b", filter=128, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res

        # XCEPTION L3-L6
        for i in range(3, 7):
            Res = X
            X = tf.nn.relu(X)
            X = separable_conv2d(X, scope="XCEPTION_{}a".format(i), filter=128, kernel_size=3,
                                 strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
            X = separable_conv2d(X, scope="XCEPTION_{}b".format(i), filter=128, kernel_size=3,
                                 strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
            X = separable_conv2d(X, scope="XCEPTION_{}c".format(i), filter=128, kernel_size=3,
                                 strides=1, padding="same", batch_norm=True, activation=None, is_training=is_training)
            X = X + Res

        # XCEPTION L7
        Res = conv2d(X, scope="CONV_res_7", filter=256,
                     kernel_size=1, strides=2, padding="same")
        X = tf.nn.relu(X)
        X = separable_conv2d(X, scope="XCEPTION_7a", filter=128, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = separable_conv2d(X, scope="XCEPTION_7b", filter=256, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res

        # XCEPTION L8-L11
        for i in range(8, 12):
            Res = X
            X = tf.nn.relu(X)
            X = separable_conv2d(X, scope="XCEPTION_{}a".format(i), filter=256, kernel_size=3,
                                 strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
            X = separable_conv2d(X, scope="XCEPTION_{}b".format(i), filter=256, kernel_size=3,
                                 strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
            X = separable_conv2d(X, scope="XCEPTION_{}c".format(i), filter=256, kernel_size=3,
                                 strides=1, padding="same", batch_norm=True, activation=None, is_training=is_training)
            X = X + Res

        # XCEPTION L12
        Res = conv2d(X, scope="CONV_res_12", filter=256,
                     kernel_size=1, strides=2, padding="same")
        X = tf.nn.relu(X)
        X = separable_conv2d(X, scope="XCEPTION_12a", filter=256, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = separable_conv2d(X, scope="XCEPTION_12b", filter=256, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res

        # XCEPTION L13-L20
        for i in range(13, 21):
            Res = X
            X = tf.nn.relu(X)
            X = separable_conv2d(X, scope="XCEPTION_{}a".format(i), filter=256, kernel_size=3,
                                 strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
            X = separable_conv2d(X, scope="XCEPTION_{}b".format(i), filter=256, kernel_size=3,
                                 strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
            X = separable_conv2d(X, scope="XCEPTION_{}c".format(i), filter=256, kernel_size=3,
                                 strides=1, padding="same", batch_norm=True, activation=None, is_training=is_training)
            X = X + Res

        # XCEPTION L21
        Res = conv2d(X, scope="CONV_res_21", filter=512,
                     kernel_size=1, strides=2, padding="same")
        X = tf.nn.relu(X)
        X = separable_conv2d(X, scope="XCEPTION_21a", filter=256, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = separable_conv2d(X, scope="XCEPTION_21b", filter=512, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res

        # XCEPTION L22
        X = separable_conv2d(X, scope="XCEPTION_22", filter=128, kernel_size=3, strides=1,
                             padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)

        # dw conv to 1x1
        X = deepwise_conv2d(X, scope="dw_conv_to_1_1", kernel_size=(
            X.shape[1], X.shape[2]), strides=1, padding="valid")

        # flatten
        X = tf.layers.flatten(X)

        # Output
        X = tf.nn.l2_normalize(X, axis=-1)
        return X


class ModelV5(TripletModel):

    def __init__(self, config):
        super().__init__(config)
        self.name = "AnonymousNet_deeper"
        self.input_size = (257, 97)

    def model(self, X, is_training, keep_prob):

        # AnonymousNet L1
        X = anonymous_net_block(input=X, filters=8, scope="AnonymousNet_1a", strides=1,
                                activate=True, batch_norm=True, is_training=is_training)
        X = anonymous_net_block(input=X, filters=8, scope="AnonymousNet_1b", strides=1,
                                activate=True, batch_norm=True, is_training=is_training)
        X = anonymous_net_block(input=X, filters=16, scope="AnonymousNet_1c", strides=2,
                                activate=True, batch_norm=True, is_training=is_training)

        # AnonymousNet L2
        X = anonymous_net_block(input=X, filters=16, scope="AnonymousNet_2a", strides=1,
                                activate=True, batch_norm=True, is_training=is_training)
        X = anonymous_net_block(input=X, filters=16, scope="AnonymousNet_2b", strides=1,
                                activate=True, batch_norm=True, is_training=is_training)
        X = anonymous_net_block(input=X, filters=32, scope="AnonymousNet_2c", strides=2,
                                activate=True, batch_norm=True, is_training=is_training)

        # AnonymousNet L3
        X = anonymous_net_block(input=X, filters=32, scope="AnonymousNet_3a", strides=1,
                                activate=True, batch_norm=True, is_training=is_training)
        X = anonymous_net_block(input=X, filters=32, scope="AnonymousNet_3b", strides=1,
                                activate=True, batch_norm=True, is_training=is_training)
        X = anonymous_net_block(input=X, filters=64, scope="AnonymousNet_3c", strides=2,
                                activate=True, batch_norm=True, is_training=is_training)

        # AnonymousNet L4
        for i in range(8):
            X = anonymous_net_block(input=X, filters=64, scope="AnonymousNet_4_{:02}".format(i), strides=1,
                                    activate=True, batch_norm=True, is_training=is_training)
        X = anonymous_net_block(input=X, filters=128, scope="AnonymousNet_4o", strides=2,
                                activate=True, batch_norm=True, is_training=is_training)

        # AnonymousNet L5
        X = anonymous_net_block(input=X, filters=128, scope="AnonymousNet_5a", strides=1,
                                activate=True, batch_norm=True, is_training=is_training)
        X = anonymous_net_block(input=X, filters=128, scope="AnonymousNet_5b", strides=1,
                                activate=True, batch_norm=True, is_training=is_training)
        X = anonymous_net_block(input=X, filters=128, scope="AnonymousNet_5c", strides=2,
                                activate=True, batch_norm=True, is_training=is_training)

        # dw conv to 1x1
        X = deepwise_conv2d(X, scope="dw_conv_to_1_1", kernel_size=(
            X.shape[1], X.shape[2]), strides=1, padding="valid")

        # flatten
        X = tf.layers.flatten(X)

        # Output
        X = tf.nn.l2_normalize(X, axis=-1)
        return X


class ModelV6(TripletModel):

    def __init__(self, config):
        super().__init__(config)
        self.name = "DenseNet"
        self.input_size = (257, 97)

    def model(self, X, is_training, keep_prob):

        # Dense L1
        X = dense_block(X, scope="DENSE_BLOCK_1a", nb_layers=4,
                        growth_rate=4, is_training=is_training)
        X = transition_layer(X, scope="TRANSITION_LAYER_1", compression=0.5,
                             pool_size=3, strides=2, padding="same", is_training=is_training)

        # Dense L2
        X = dense_block(X, scope="DENSE_BLOCK_2a", nb_layers=4,
                        growth_rate=8, is_training=is_training)
        X = transition_layer(X, scope="TRANSITION_LAYER_2", compression=0.5,
                             pool_size=3, strides=2, padding="same", is_training=is_training)

        # Dense L3
        X = dense_block(X, scope="DENSE_BLOCK_3a", nb_layers=16,
                        growth_rate=8, is_training=is_training)
        X = transition_layer(X, scope="TRANSITION_LAYER_3", compression=0.5,
                             pool_size=3, strides=2, padding="same", is_training=is_training)

        # Dense L4
        X = dense_block(X, scope="DENSE_BLOCK_4a", nb_layers=16,
                        growth_rate=16, is_training=is_training)
        X = dense_block(X, scope="DENSE_BLOCK_4b", nb_layers=16,
                        growth_rate=16, is_training=is_training)
        X = transition_layer(X, scope="TRANSITION_LAYER_4", compression=0.5,
                             pool_size=3, strides=2, padding="same", is_training=is_training)

        # Dense L5
        X = dense_block(X, scope="DENSE_BLOCK_5a", nb_layers=8,
                        growth_rate=32, is_training=is_training)
        X = dense_block(X, scope="DENSE_BLOCK_5b", nb_layers=8,
                        growth_rate=32, is_training=is_training)
        X = transition_layer(X, scope="TRANSITION_LAYER_5", compression=0.5,
                             pool_size=3, strides=2, padding="same", is_training=is_training)

        # dw conv to 1x1
        X = deepwise_conv2d(X, scope="dw_conv_to_1_1", kernel_size=(
            X.shape[1], X.shape[2]), strides=1, padding="valid")

        # flatten
        X = tf.layers.flatten(X)

        # Output
        X = tf.nn.l2_normalize(X, axis=-1)
        return X


class Model(ModelV6):
    def __init__(self, config):
        super().__init__(config)
