import numpy as np
from PIL import Image
import torch as t
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer


class BBProposer():
    def __init__(self, opt) -> None:
        self.FasterRCNN = FasterRCNNVGG16()
        self.trainer = FasterRCNNTrainer(self.FasterRCNN).cuda()
        self.opt = opt
        self.trainer.load(
            '/home/tim/文档/Pytorch_ws/robotcv_pipeline/checkpoints/fasterrcnn_11252310_0.9529958677685952')
        # this model was trained from torchvision-pretrained model
        self.opt.caffe_pretrain = False

    def predict(self, img):
        newimg = self.convert(img)
        tempimg = t.from_numpy(newimg)[None]
        _bboxes, _labels, _scores = self.trainer.faster_rcnn.predict(
            tempimg, visualize=True)
        return _bboxes, _labels, _scores

    def convert(self, imgcpy: Image.Image, dtype=np.float32, color=True):
        try:
            if color:
                img = imgcpy.convert('RGB')
            else:
                img = imgcpy.convert('P')
            img = np.asarray(img, dtype=dtype)
        finally:
            # print("successfuly convert img!")
            pass

        if img.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            return img[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            return img.transpose((2, 0, 1))
