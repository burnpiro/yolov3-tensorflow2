import tensorflow as tf


class BatchNorm(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        # if training not set then set it to be tf.constant
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)
