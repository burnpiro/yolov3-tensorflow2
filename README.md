## YOLOv3 implementation in TensorFlow 2.0

![alt text][image]

#### Installation

```bash
pip3 install -r ./requirements.txt
wget https://pjreddie.com/media/files/yolov3.weights -O ./yolov3.weights
```

#### Detect image

```bash
python detect.py --image ./image.jpg --output ./output.jpg
```


#### Detect video

```bash
python detect_video.py --video ./video.mp4
```

```bash
# for webcam
python detect_video.py --video 0
```

#### Papers and thanks

- [YOLO website](https://pjreddie.com/darknet/yolo/)
- [YOLOv3 paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [NMS paper](https://arxiv.org/pdf/1704.04503.pdf)
- [NMS implementation](https://github.com/bharatsingh430/soft-nms)
- [GIOU Paper](https://giou.stanford.edu/GIoU.pdf)
- [DarkNet Implementation](https://github.com/pjreddie/darknet)
- [YOLO implementation](https://github.com/zzh8829/yolov3-tf2)


[image]: ./default_output.jpg "Logo Title Text 2"