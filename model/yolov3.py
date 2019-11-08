import tensorflow as tf
from model.darknet53 import darknet53
from model.utils.convolutional import convolutional
from model.utils.upsample import upsample
from config import cfg
from utils import read_class_names

NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))


def YOLOv3(input_layer):
    route_1, route_2, conv = darknet53(input_layer)

    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))

    conv_lobj_branch = convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3*(NUM_CLASS + 5)), activate=False, batch_norm=False)

    conv = convolutional(conv, (1, 1,  512,  256))
    conv = upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = convolutional(conv, (1, 1, 768, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, batch_norm=False)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = convolutional(conv, (1, 1, 384, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = convolutional(conv_sobj_branch, (1, 1, 256, 3*(NUM_CLASS +5)), activate=False, batch_norm=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]
