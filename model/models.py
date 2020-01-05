import tensorflow as tf

from model.triplet_model import TripletModel


class ModelV1(TripletModel):

    def __init__(self, config):
        super().__init__(config)
        self.name = "inception"

    def model(self, X, is_training, keep_prob):
        # mask = (X + 1) / 2

        # CONV L1
        X = self.conv2d(X, scope="CONV_1", filter=32, kernel_size=3, strides=1,
                        padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        # mask = tf.layers.average_pooling2d(mask, pool_size=3, strides=2, padding="same")
        # mask = tf.where(mask>0.1, mask, tf.zeros_like(mask))
        # X = tf.cast(tf.greater(mask, 0), tf.float32) * X

        # INCEPTION L2
        Res = self.conv2d(X, scope="CONV_res_2", filter=64,
                          kernel_size=1, strides=2, padding="same")
        X = self.inception_v2(input=X, scope="INCEPTION_2a", filters=64,
                              batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.inception_v2(input=X, scope="INCEPTION_2b", filters=64,
                              batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        # mask = tf.layers.max_pooling2d(mask, pool_size=3, strides=2, padding="same")
        # X = tf.cast(tf.greater(mask, 0), tf.float32) * X
        X = X + Res

        # INCEPTION L3
        Res = self.conv2d(X, scope="CONV_res_3", filter=128,
                          kernel_size=1, strides=2, padding="same")
        X = tf.nn.relu(X)
        X = self.inception_v2(input=X, scope="INCEPTION_3a", filters=128,
                              batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.inception_v2(input=X, scope="INCEPTION_3b", filters=128,
                              batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        # mask = tf.layers.max_pooling2d(mask, pool_size=3, strides=2, padding="same")
        # X = tf.cast(tf.greater(mask, 0), tf.float32) * X
        X = X + Res

        # INCEPTION L4
        Res = self.conv2d(X, scope="CONV_res_4", filter=256,
                          kernel_size=1, strides=2, padding="same")
        X = tf.nn.relu(X)
        X = self.inception_v2(input=X, scope="INCEPTION_4a", filters=256,
                              batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.inception_v2(input=X, scope="INCEPTION_4b", filters=256,
                              batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.inception_v2(input=X, scope="INCEPTION_4c", filters=256,
                              batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res
        X = self.maxout(X, num_units=128)

        # dw conv
        X = self.deepwise_conv2d(X, scope="dw_conv", kernel_size=(
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

    def model(self, X, is_training, keep_prob):

        # CONV L1
        X = self.conv2d(X, scope="CONV_1", filter=32, kernel_size=3, strides=1,
                        padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")

        # XCEPTION L2
        Res = self.conv2d(X, scope="CONV_res_2", filter=128,
                          kernel_size=1, strides=2, padding="same")
        X = self.separable_conv2d(X, scope="XCEPTION_2a", filter=64, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.separable_conv2d(X, scope="XCEPTION_2b", filter=128, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res

        # XCEPTION L3
        Res = X
        X = tf.nn.relu(X)
        X = self.separable_conv2d(X, scope="XCEPTION_3a", filter=128, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.separable_conv2d(X, scope="XCEPTION_3b", filter=128, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.separable_conv2d(X, scope="XCEPTION_3c", filter=128, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = X + Res

        # XCEPTION L4
        Res = self.conv2d(X, scope="CONV_res_4", filter=128,
                          kernel_size=1, strides=2, padding="same")
        X = tf.nn.relu(X)
        X = self.separable_conv2d(X, scope="XCEPTION_4a", filter=128, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.separable_conv2d(X, scope="XCEPTION_4b", filter=128, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res

        # XCEPTION L5
        Res = self.conv2d(X, scope="CONV_res_5", filter=256,
                          kernel_size=1, strides=2, padding="same")
        X = tf.nn.relu(X)
        X = self.separable_conv2d(X, scope="XCEPTION_5a", filter=256, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.separable_conv2d(X, scope="XCEPTION_5b", filter=256, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res
        X = self.maxout(X, num_units=128)

        # dw conv to 1x1
        X = self.deepwise_conv2d(X, scope="dw_conv_to_1_1", kernel_size=(
            X.shape[1], X.shape[2]), strides=1, padding="valid")

        # flatten
        X = tf.layers.flatten(X)

        # Output
        X = tf.nn.l2_normalize(X, axis=-1)
        return X


class ModelV3(TripletModel):

    def __init__(self, config):
        super().__init__(config)
        self.name = "Nnet_v1"

    def model(self, X, is_training, keep_prob):
        # mask = (X + 1) / 2

        # Nnet L1
        X = self.nnet_v1(input=X, filters=16, scope="Nnet_1", strides=2,
                         activate=True, batch_norm=True, is_training=is_training)
        # mask = tf.layers.average_pooling2d(mask, pool_size=3, strides=2, padding="same")
        # mask = tf.where(mask>0.1, mask, tf.zeros_like(mask))
        # X = tf.cast(tf.greater(mask, 0), tf.float32) * X

        # Nnet L2
        X = self.nnet_v1(input=X, filters=16, scope="Nnet_2a", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=32, scope="Nnet_2b", strides=2,
                         activate=True, batch_norm=True, is_training=is_training)
        # mask = tf.layers.max_pooling2d(mask, pool_size=3, strides=2, padding="same")
        # X = tf.cast(tf.greater(mask, 0), tf.float32) * X

        # Nnet L3
        X = self.nnet_v1(input=X, filters=32, scope="Nnet_3a", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=64, scope="Nnet_3b", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)

        # Nnet L4
        X = self.nnet_v1(input=X, filters=64, scope="Nnet_4a", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_4b", strides=2,
                         activate=True, batch_norm=True, is_training=is_training)
        # mask = tf.layers.max_pooling2d(mask, pool_size=3, strides=2, padding="same")
        # X = tf.cast(tf.greater(mask, 0), tf.float32) * X

        # Nnet L5
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_5a", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_5b", strides=2,
                         activate=True, batch_norm=True, is_training=is_training)
        # mask = tf.layers.max_pooling2d(mask, pool_size=3, strides=2, padding="same")
        # X = tf.cast(tf.greater(mask, 0), tf.float32) * X

        # dw conv to 1x1
        X = self.deepwise_conv2d(X, scope="dw_conv_to_1_1", kernel_size=(
            X.shape[1], X.shape[2]), strides=1, padding="valid")

        # flatten
        X = tf.layers.flatten(X)

        # Output
        X = tf.nn.l2_normalize(X, axis=-1)
        return X


class ModelV4(TripletModel):
    """ 图片大小 97 * 257 """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Xception_deeper"

    def model(self, X, is_training, keep_prob):

        # CONV L1
        X = self.conv2d(X, scope="CONV_1", filter=16, kernel_size=3, strides=1,
                        padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")

        # XCEPTION L2
        Res = self.conv2d(X, scope="CONV_res_2", filter=128,
                          kernel_size=1, strides=2, padding="same")
        X = self.separable_conv2d(X, scope="XCEPTION_2a", filter=64, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.separable_conv2d(X, scope="XCEPTION_2b", filter=128, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res

        # XCEPTION L3-L6
        for i in range(3, 7):
            Res = X
            X = tf.nn.relu(X)
            X = self.separable_conv2d(X, scope="XCEPTION_{}a".format(i), filter=128, kernel_size=3,
                                      strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
            X = self.separable_conv2d(X, scope="XCEPTION_{}b".format(i), filter=128, kernel_size=3,
                                      strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
            X = self.separable_conv2d(X, scope="XCEPTION_{}c".format(i), filter=128, kernel_size=3,
                                      strides=1, padding="same", batch_norm=True, activation=None, is_training=is_training)
            X = X + Res

        # XCEPTION L7
        Res = self.conv2d(X, scope="CONV_res_7", filter=256,
                          kernel_size=1, strides=2, padding="same")
        X = tf.nn.relu(X)
        X = self.separable_conv2d(X, scope="XCEPTION_7a", filter=128, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.separable_conv2d(X, scope="XCEPTION_7b", filter=256, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res

        # XCEPTION L8-L11
        for i in range(8, 12):
            Res = X
            X = tf.nn.relu(X)
            X = self.separable_conv2d(X, scope="XCEPTION_{}a".format(i), filter=256, kernel_size=3,
                                      strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
            X = self.separable_conv2d(X, scope="XCEPTION_{}b".format(i), filter=256, kernel_size=3,
                                      strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
            X = self.separable_conv2d(X, scope="XCEPTION_{}c".format(i), filter=256, kernel_size=3,
                                      strides=1, padding="same", batch_norm=True, activation=None, is_training=is_training)
            X = X + Res

        # XCEPTION L12
        Res = self.conv2d(X, scope="CONV_res_12", filter=256,
                          kernel_size=1, strides=2, padding="same")
        X = tf.nn.relu(X)
        X = self.separable_conv2d(X, scope="XCEPTION_12a", filter=256, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.separable_conv2d(X, scope="XCEPTION_12b", filter=256, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res

        # XCEPTION L13-L20
        for i in range(13, 21):
            Res = X
            X = tf.nn.relu(X)
            X = self.separable_conv2d(X, scope="XCEPTION_{}a".format(i), filter=256, kernel_size=3,
                                      strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
            X = self.separable_conv2d(X, scope="XCEPTION_{}b".format(i), filter=256, kernel_size=3,
                                      strides=1, padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
            X = self.separable_conv2d(X, scope="XCEPTION_{}c".format(i), filter=256, kernel_size=3,
                                      strides=1, padding="same", batch_norm=True, activation=None, is_training=is_training)
            X = X + Res

        # XCEPTION L21
        Res = self.conv2d(X, scope="CONV_res_21", filter=512,
                          kernel_size=1, strides=2, padding="same")
        X = tf.nn.relu(X)
        X = self.separable_conv2d(X, scope="XCEPTION_21a", filter=256, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)
        X = self.separable_conv2d(X, scope="XCEPTION_21b", filter=512, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=None, is_training=is_training)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, padding="same")
        X = X + Res

        # XCEPTION L22
        X = self.separable_conv2d(X, scope="XCEPTION_22", filter=128, kernel_size=3, strides=1,
                                  padding="same", batch_norm=True, activation=tf.nn.relu, is_training=is_training)

        # dw conv to 1x1
        X = self.deepwise_conv2d(X, scope="dw_conv_to_1_1", kernel_size=(
            X.shape[1], X.shape[2]), strides=1, padding="valid")

        # flatten
        X = tf.layers.flatten(X)

        # Output
        X = tf.nn.l2_normalize(X, axis=-1)
        return X


class ModelV5(TripletModel):
    """ 图片大小 97 * 257 """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Nnet_v1_deeper"

    def model(self, X, is_training, keep_prob):
        # mask = (X + 1) / 2

        # Nnet L1
        X = self.nnet_v1(input=X, filters=16, scope="Nnet_1", strides=2,
                         activate=True, batch_norm=True, is_training=is_training)

        # Nnet L2
        X = self.nnet_v1(input=X, filters=16, scope="Nnet_2a", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=32, scope="Nnet_2b", strides=2,
                         activate=True, batch_norm=True, is_training=is_training)

        # Nnet L3
        X = self.nnet_v1(input=X, filters=64, scope="Nnet_3a", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=64, scope="Nnet_3b", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=64, scope="Nnet_3c", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_3d", strides=2,
                         activate=True, batch_norm=True, is_training=is_training)

        # Nnet L4
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_4a", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_4b", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_4c", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        # X = self.nnet_v1(input=X, filters=128, scope="Nnet_4d", strides=2,
        #                  activate=True, batch_norm=True, is_training=is_training)

        # Nnet L5
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_5a", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_5b", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_5c", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_5d", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_5e", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_5f", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)

        # Nnet L6
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_6a", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_6b", strides=2,
                         activate=True, batch_norm=True, is_training=is_training)

        # Nnet L7
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_7a", strides=1,
                         activate=True, batch_norm=True, is_training=is_training)
        X = self.nnet_v1(input=X, filters=128, scope="Nnet_7b", strides=2,
                         activate=True, batch_norm=True, is_training=is_training)

        # dw conv to 1x1
        X = self.deepwise_conv2d(X, scope="dw_conv_to_1_1", kernel_size=(
            X.shape[1], X.shape[2]), strides=1, padding="valid")

        # flatten
        X = tf.layers.flatten(X)

        # Output
        X = tf.nn.l2_normalize(X, axis=-1)
        return X


class Model(ModelV3):
    def __init__(self, config):
        super().__init__(config)
