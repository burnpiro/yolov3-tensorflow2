import tensorflow as tf
from model.utils.batch_norm import BatchNorm


def convolutional(input_layer, filters_shape, down_sample=False, activate=True, batch_norm=True,
                  regularization=0.0005, reg_stddev=0.01, activate_alpha=0.1):
    if down_sample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1
    conv = tf.keras.layers.Conv2D(
        filters=filters_shape[-1],
        kernel_size=filters_shape[0],
        strides=strides,
        padding=padding,
        use_bias=not batch_norm,  # use bias only when not normalizing
        kernel_regularizer=tf.keras.regularizers.l2(regularization),
        kernel_initializer=tf.random_normal_initializer(stddev=reg_stddev),
        bias_initializer=tf.constant_initializer(0.)
    )(input_layer)

    if batch_norm:
        conv = BatchNorm()(conv)
    if activate:
        conv = tf.nn.leaky_relu(conv, alpha=activate_alpha)

    return conv
