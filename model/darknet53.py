import tensorflow as tf
from model.utils.convolutional import convolutional
from model.utils.res_block import res_block


def darknet53(input_data):
    input_data = convolutional(input_data, (3, 3, 3, 32))
    input_data = convolutional(input_data, (3, 3, 32, 64), down_sample=True)

    for i in range(1):
        input_data = res_block(input_data, 64, 32, 64)

    input_data = convolutional(input_data, (3, 3, 64, 128), down_sample=True)

    for i in range(2):
        input_data = res_block(input_data, 128, 64, 128)

    input_data = convolutional(input_data, (3, 3, 128, 256), down_sample=True)

    for i in range(8):
        input_data = res_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), down_sample=True)

    for i in range(1):
        input_data = res_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), down_sample=True)

    for i in range(1):
        input_data = res_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data
