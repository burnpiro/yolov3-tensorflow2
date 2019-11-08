from easydict import EasyDict

__C = EasyDict()

cfg = __C

# create YOLO dict
__C.YOLO = EasyDict()

__C.YOLO.CLASSES = "./data/classes.names"
__C.YOLO.ANCHORS = "./data/anchors.txt"
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5

# create Train options dict
__C.TRAIN = EasyDict()
__C.TRAIN.ANNOTATION_PATH = "./data/train.txt"
__C.TRAIN.BATCH_SIZE = 4
__C.TRAIN.INPUT_SIZE = [416]  # [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.DATA_AUG = True
__C.TRAIN.LR_INIT = 1e-3
__C.TRAIN.LR_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 2
__C.TRAIN.EPOCHS = 30

# create TEST options dict
__C.TEST = EasyDict()
__C.TEST.ANNOTATION_PATH = "./data/test.txt"
__C.TEST.BATCH_SIZE = 2
__C.TEST.INPUT_SIZE = 544  # 320, 352, 384, 416, 448, 480, 512, 544, 576, 608
__C.TEST.DATA_AUG = False
__C.TEST.DETECTED_IMAGE_PATH = "./output/detection/"
__C.TEST.SCORE_THRESHOLD = 0.3
__C.TEST.IOU_THRESHOLD = 0.45
