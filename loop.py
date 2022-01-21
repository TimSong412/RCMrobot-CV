# from PIL import Image
# from propose import BBProposer
from circlepropose import Circle
import cv2
import numpy as np
from matting import BcRd
import time
import network

DEBUG = True


class Loop():
    def __init__(self) -> None:
        self.cam = cv2.VideoCapture(4)
        self.Predictor = Circle()
        ret, self.preimg = self.cam.read()
        if DEBUG:
            cv2.imshow("ini_img", self.preimg)
            cv2.waitKey(0)
        if ret:
            self.pre_gray, self.pre_mask, self.pre_center = self.imgprocess(
                self.preimg)
            self.boarder = np.zeros(
                (self.preimg.shape[0], self.preimg.shape[1]), dtype=np.uint8)
            self.boarder[10:self.preimg.shape[0] -
                         10, 10:self.preimg.shape[1]-10] += 1
        else:
            print("Error! can not open camera!")
            exit()

    def imgprocess(self, img):
        bb = self.Predictor.predict(img)
        bbox = np.array(bb[0][0], dtype=int)
        bg = BcRd(img)
        mask = bg.GenMask(bbox)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        totalmask = np.zeros(gray.shape[0:2], dtype=np.uint8)
        # totalmask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = mask
        totalmask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1
        if DEBUG:
            cv2.imshow("mask", totalmask*250)
        # (y, x)
        return gray, totalmask, np.array([(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2])

    def refresh(self, newimg, newgray, newmask):
        self.preimg = newimg.copy()
        self.pre_gray = newgray.copy()
        self.pre_mask = newmask.copy()

    def norm(self, vec):
        if abs(vec[0]) > 0.01 or abs(vec[1]) > 0.01:
            deno = max(abs(vec[0]), abs(vec[1]))*100
        else:
            deno = 1
        return vec/deno


if __name__ == '__main__':
    sender = network.Sender('192.168.137.216')
    mainloop = Loop()
    while True:
        st = time.time()
        for i in range(20):
            ret, tmpimg = mainloop.cam.read()
            cv2.waitKey(20)
        ret, newimg = mainloop.cam.read()
        ed = time.time()
        print("cam= ", ed-st)
        if not ret:
            print("CAMERA ERROR!")
            break
        newgray, newmask, newcenter = mainloop.imgprocess(newimg)
        if DEBUG:
            overlap = np.array(mainloop.pre_gray*0.3 +
                               newgray*0.7, dtype=np.uint8)
            cv2.imshow("overlap", overlap)

        flow = cv2.calcOpticalFlowFarneback(
            mainloop.pre_gray, newgray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow *= np.expand_dims(mainloop.pre_mask, 2).repeat(2, axis=2)
        flow *= np.expand_dims(mainloop.boarder, 2).repeat(2, axis=2)
        flowdata = flow.reshape(1, -1, 2)[0]
        shift = np.mean(flowdata, axis=0)  # (y_width, x_height)
        move = (newcenter - mainloop.pre_center)/10000
        shift /= 100
        print("shift= ", shift)
        print("move= ", move)

        # nomalization
        shift = mainloop.norm(shift)
        move = mainloop.norm(move)
        if abs(shift[0]) < 0.005 and abs(shift[1]) < 0.005:
            if abs(move[0]) > 0.004 or abs(move[1]) > 0.004:
                print("MOVE!")
                sender.sendvec(move[0], move[1])
            # mainloop.refresh(newimg, newgray, newmask)
                pass
            else:
                print("CONVERGE!")
        else:
            print("SHIFT!")
            sender.sendvec(shift[0], shift[1])
            pass

        if cv2.waitKey(400) == 115:
            break
        ed = time.time()
        print("t=", ed-st, "--------------------------")
