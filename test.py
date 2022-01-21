import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
frame = cv2.imread("/home/tim/文档/Pytorch_ws/fasterrcnn_robot/ImgAnno/Anno/1-2/JPGImages/000860.jpg", cv2.IMREAD_COLOR)
# setup initial location of window
r, h, c, w = 0, 212, 637, 1217  # simply hardcoded the values
track_window = (c, r, w, h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                   np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

# apply meanshift to get the new location
ret, track_window = cv2.CamShift(dst, track_window, term_crit)

# Draw it on image
pts = cv2.boxPoints(ret)
pts = np.int0(pts)
img2 = cv2.polylines(frame, [pts], True, 255, 2)
cv2.imshow('img2', img2)

k = cv2.waitKey(0)
# if k == 27:
#     break
# else:
#     cv2.imwrite(chr(k)+".jpg", img2)

cv2.destroyAllWindows()
'''
region1 = cv2.imread("/home/tim/文档/Pytorch_ws/robotcv_pipeline/region3.png")
region2 = cv2.imread("/home/tim/文档/Pytorch_ws/robotcv_pipeline/region4.png")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

rg1data2 = np.array(region1.reshape(-1, 3), dtype=np.float32)
ret2 = cv2.kmeans(rg1data2, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
mask2 = ret2[1].reshape(region1.shape[0:2])
rg1data = np.zeros((len(region1)*len(region1[0]), 5), dtype=np.float32)
cols = len(region1[0])
for i in range(len(region1)):
    for j in range(cols):
        rg1data[i*cols+j][0:2] = [i/8, j/8]
        rg1data[i*cols+j][2:5] = region1[i, j]
# print(rg1data)

retval = cv2.kmeans(rg1data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

mask = retval[1].reshape(region1.shape[0:2])
cv2.imshow("ori", region1)
cv2.imshow("mask", np.array(mask*80, dtype=np.uint8))
cv2.imshow("mask2", np.array(mask2*80, dtype=np.uint8))
cv2.waitKey(0)
