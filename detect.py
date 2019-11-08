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

    img = cv2.imread(FLAGS.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = img.shape[:2]

    image_data = utils.image_preporcess(np.copy(img), [FLAGS.size, FLAGS.size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    # model.summary()
    utils.load_weights(model, FLAGS.weights)

    pred_bbox = model.predict(image_data)
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