# from PIL import Image
# from propose import BBProposer
import threading
import cv2
from detector import Detector
import time
import network

DEBUG = True
lastimg = None
end = False
CAMIDX = 4


class cap(threading.Thread):
    def __init__(self):
        global lastimg
        threading.Thread.__init__(self)
        self.camera = cv2.VideoCapture(CAMIDX)
        ret, lastimg = self.camera.read()
        print("INIT")
        if not ret:
            print("READ IMG ERROR")

    def run(self):
        global lastimg
        while True:
            ret, lastimg = self.camera.read()
            if ret:
                time.sleep(0.003)
            else:
                print("CAM_ERROR")
            if end:
                break


if __name__ == '__main__':
    sender = network.Sender('192.168.238.158')
    cam = cap()
    cam.start()
    detector = Detector(classes=[0, 1])
    out = cv2.VideoWriter(
        './result/demo5.mp4', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10, (640, 480), True)
    log = open("./result/center5.txt", 'w')
    while True:
        newimg = lastimg.copy()
        cv2.imshow("monitor", newimg)
        if cv2.waitKey(30) == 113:
            break
    cv2.destroyAllWindows()

    while not end:
        st = time.time()
        newimg = lastimg.copy()
        ed = time.time()
        cx, cy, im = detector.detect(newimg, 0)
        out.write(im)
        log.write(str(cx)+" "+str(cy)+"\n")
        # cv2.waitKey(0)
        if cx > 0:
            sender.sendvec(cx, cy)
            pass
        print("center= ", cx, cy)

        if cv2.waitKey(50) == 113:
            end = True
            break
        ed = time.time()
        print("t=", ed-st, "--------------------------")
    cam.join()
    log.close()
    out.release()
