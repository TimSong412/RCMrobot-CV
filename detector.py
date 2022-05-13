# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors
from utils.general import (
    LOGGER, check_file, check_img_size, non_max_suppression, print_args, scale_coords)
from utils.datasets import IMG_FORMATS, VID_FORMATS
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Detector():
    @torch.no_grad()
    def __init__(self,
                 weights=ROOT / 'best.pt',  # model.pt path(s)
                 data=ROOT / 'data.yaml',  # dataset.yaml path
                 imgsz=(640, 640),  # inference size (height, width)
                 conf_thres=0.4,  # confidence threshold
                 iou_thres=0.4,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 visualize=False,  # visualize features
                 line_thickness=3,  # bounding box thickness (pixels)
                 hide_labels=False,  # hide labels
                 hide_conf=False,  # hide confidences
                 half=False,  # use FP16 half-precision inference
                 dnn=False,  # use OpenCV DNN for ONNX inference) -> None:
                 ):
        self.device = select_device(device)
        self.model = DetectMultiBackend(
            weights, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.bs = 1  # batch_size

        self.conf_thres = conf_thres  # confidence threshold
        self.iou_thres = iou_thres  # NMS IOU threshold
        self.classes = classes  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = agnostic_nms  # class-agnostic NMS
        self.augment = augment  # augmented inference
        self.visualize = visualize  # visualize features
        self.max_det = max_det
        # bounding box thickness (pixels)
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels  # hide labels
        self.hide_conf = hide_conf  # hide confidences
        half = False,  # use FP16 half-precision inference
        dnn = False,  # use OpenCV DNN for ONNX inference) -> None:

    @torch.no_grad()
    def detect(self, img0, target=0, visual=True):
        im0 = img0.copy()
        im = letterbox(img0, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        # Run inference
        self.model.warmup(
            imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0

        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        cx = -1
        cy = -1
        find = False
        print("target= ", target)
        for i, det in enumerate(pred):  # per image
            seen += 1
            # normalization gain whwh
            if visual:
                annotator = Annotator(
                    im0, line_width=self.line_thickness, example=str(self.names))
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], im0.shape).round()
            if len(det):
                # Rescale boxes from img_size to im0 size
                # Write results
                maxid = -1
                maxconf = 0
                print("det= ", det)
                for i, bbox in enumerate(det):
                    if int(bbox[5]) == target and bbox[4] > maxconf:
                        find = True
                        maxconf = bbox[4].item()
                        maxid = i
                if maxid < 0:
                    continue
                *xyxy, conf, cls = det[maxid]
                print("cls= ", cls)
                c = int(cls)  # integer class
                if visual:
                    label = None if self.hide_labels else (
                        self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                cx = (xyxy[0].item()+xyxy[2].item())/2
                cy = (xyxy[1].item()+xyxy[3].item())/2

                # for *xyxy, conf, cls in reversed(det):
                #     c = int(cls)  # integer class
                #     if c != target:
                #         continue
                #     if visual:
                #         label = None if self.hide_labels else (
                #             self. names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                #         annotator.box_label(xyxy, label, color=colors(c, True))
                #     cx = (xyxy[0].item()+xyxy[2].item())/2
                #     cy = (xyxy[1].item()+xyxy[3].item())/2
            else:
                print("NONE DETECTED!")

            # Stream results
        if visual:
            if find:
                im0 = annotator.result()
            cv2.imshow("img0", im0)
        return cx, cy, im0


if __name__ == "__main__":
    # opt = parse_opt()
    # main(opt)
    detector = Detector()
    img = cv2.imread("./img0.jpg")
    cx, cy, im = detector.detect(img, target=1)
    cv2.waitKey(0)
    print("cx, cy= ", cx, cy)
