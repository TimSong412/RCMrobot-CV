import cv2
import numpy as np


img = cv2.imread(
    "/home/tim/文档/Pytorch_ws/robotcv_pipeline/video/1.png", cv2.IMREAD_COLOR)


retval, corners = cv2.findChessboardCornersSB(img, (8, 5))
cv2.drawChessboardCorners(img, (8, 5), corners, retval)

cv2.imshow("res", img)
cv2.waitKey(0)

objpts = []
blk = 29.1
for i in range(5):
    for j in range(8):
        objpts.append([j*blk, i*blk, 0])
objs = np.array(objpts, dtype=np.float32)
# print(objs)

InM = np.ndarray((3, 3))
CoeM = np.ndarray((4, 1))

print(img.shape[1], img.shape[0])

cv2.calibrateCamera([objs], [corners], (img.shape[1], img.shape[0]), InM, CoeM)

print(InM)
print(CoeM)

# cam = cv2.VideoCapture(4)
# ret, frame = cam.read()
# cv2.imshow("cam", frame)
# if cv2.waitKey(0) == 115:
#     cv2.imwrite("/home/tim/文档/Pytorch_ws/robotcv_pipeline/cam1.png", frame)
# cam.release()
