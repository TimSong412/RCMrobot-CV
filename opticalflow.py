import cv2
import numpy as np
import matplotlib.pyplot as plt


# region1 = cv2.imread(
#     "/home/tim/文档/Pytorch_ws/fasterrcnn_robot/ImgAnno/Anno/1-2/JPGImages/000027.jpg")
# region2 = cv2.imread(
#     "/home/tim/文档/Pytorch_ws/fasterrcnn_robot/ImgAnno/Anno/1-2/JPGImages/000028.jpg")

img1 = cv2.imread("/home/tim/文档/Pytorch_ws/robotcv_pipeline/0.jpg")
img2 = cv2.imread("/home/tim/文档/Pytorch_ws/robotcv_pipeline/1.jpg")
print(img1.shape)
region1 = cv2.resize(img1, np.array([0.2*img1.shape[1], 0.2*img1.shape[0]], dtype=int))
region2 = cv2.resize(img2, np.array([0.2*img2.shape[1], 0.2*img2.shape[0]], dtype=int))

gray1 = cv2.cvtColor(region1, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(region2, cv2.COLOR_RGB2GRAY)

cv2.imshow("overlap", np.array(gray1*0.3+gray2*0.7, dtype=np.uint8))

# cv2.imshow("img1", region1)
# # cv2.imshow("resize", cv2.resize(region2, (region1.shape[1], region1.shape[0])))
# cv2.imshow("crop1", region1[0:region2.shape[0], 0:region2.shape[1], :])
# cv2.imshow("crop2", region2[0:region1.shape[0], 0:region1.shape[1], :])
# cv2.waitKey(0)

hsv = np.zeros_like(region1)
hsv[..., 1] = 255
flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2, None, 0.5, 3, 15, 3, 8, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
print(flow.shape)
# hsv[..., 0] = ang*180/np.pi/2
# hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
# bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# cv2.imshow('frame2', bgr)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
rg1data = np.array(flow.reshape(-1, 2), dtype=np.float32)
ret1 = cv2.kmeans(rg1data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
mask1 = ret1[1].reshape(flow.shape[0:2])
cv2.imshow("mask1", np.array(mask1*80, dtype=np.uint8))

# U = flow[:, :, 0].reshape(-1)*100
# V = flow[:, :, 1].reshape(-1)*100
U = flow[599:602, 399:402, 0]*10
V = flow[599:602, 399:402, 1]*10
print("flow", flow[600, 400, :])
print("U ", U.shape, "V ", V.shape)
C = np.sqrt(U**2+V**2)
# X, Y = np.mgrid[0: region1.shape[0], 0: region1.shape[1]]

X, Y = np.mgrid[0:3, 0:3]
print(X.shape) 
print(U.shape)
cv2.waitKey(0)
plt.quiver(Y, X[:: -1], V, -U, C)
# plt.quiver(Y, X[::-1], Y+1, X[::-1]+1, Y+X[::-1])
plt.show()