import tensorflow as tf


def bbox_iou(boxes1, boxes2):
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

    return 1.0 * is_area / union_area
