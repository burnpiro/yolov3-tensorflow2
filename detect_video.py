import cv2
import numpy as np
import utils as utils
import tensorflow as tf
import time
from absl import app, flags, logging
from absl.flags import FLAGS
from model.yolov3 import YOLOv3
from model.decode_output import decode

flags.DEFINE_string('weights', './yolov3.weights',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './video.mp4', 'path to input image')
flags.DEFINE_string('output', './output.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')


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

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))


    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        img_size = img.shape[:2]
        img_in = tf.expand_dims(img, 0)
        img_in = utils.transform_images(img_in, FLAGS.size)

        t1 = time.time()
        pred_bbox = model.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        boxes = utils.postprocess_boxes(pred_bbox, img_size, FLAGS.size, 0.3)
        boxes = utils.nms(boxes, 0.45, method='nms')
        img = utils.draw_outputs(img, boxes)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass