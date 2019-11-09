import cv2
import numpy as np
import utils as utils
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from model.yolov3 import YOLOv3
from model.decode_output import decode

flags.DEFINE_string('weights', './yolov3.weights',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './kite.jpg', 'path to input image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')


def main(_argv):
    input_layer = tf.keras.layers.Input([FLAGS.size, FLAGS.size, 3])
    feature_maps = YOLOv3(input_layer)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    # model.summary()
    utils.load_weights(model, FLAGS.weights)

    test_img = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
    img_size = test_img.shape[:2]
    test_img = tf.expand_dims(test_img, 0)
    test_img = utils.transform_images(test_img, FLAGS.size)

    pred_bbox = model.predict(test_img)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    boxes = utils.postprocess_boxes(pred_bbox, img_size, FLAGS.size, 0.3)
    boxes = utils.nms(boxes, 0.45, method='nms')

    original_image = cv2.imread(FLAGS.image)
    img = utils.draw_outputs(original_image, boxes)
    cv2.imwrite(FLAGS.output, img)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass