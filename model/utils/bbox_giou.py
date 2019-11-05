# IoU improvement base on paper "Generalized Intersection over Union"
# https://giou.stanford.edu/GIoU.pdf
# authors: Rezatofighi, Hamid and Tsoi, Nathan and Gwak, JunYoung and Sadeghian, Amir and Reid, Ian and Savarese, Silvio

import tensorflow as tf


def bbox_giou(boxes1, boxes2):
    box1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                      boxes1[..., :2] + boxes1[..., 2:] * 0.5],
                     axis=-1)
    box2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                      boxes2[..., :2] + boxes2[..., 2:] * 0.5],
                     axis=-1)

    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2], - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., :2], boxes2[..., 2:])

    intersection = tf.maximum(right_down - left_up, 0.0)
    is_area = intersection[..., 0] * intersection[..., 1]
    union_area = area1 + area2 - is_area

    iou = 1.0 * is_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., :2])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]

    return iou - 1.0 * (enclose_area - union_area) / enclose_area
