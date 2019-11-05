from model.utils.convolutional import convolutional


def res_block(input_layer, input_channel, filter_num1, filter_num2):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2))

    res_output = short_cut + conv
    return res_output
