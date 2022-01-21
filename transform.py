import cv2
import numpy as np
import matplotlib.pyplot as plt


img0 = cv2.imread(
    "/home/tim/文档/Pytorch_ws/robotcv_pipeline/imgpairs/img10.jpg", cv2.IMREAD_COLOR)
img1 = cv2.imread(
    "/home/tim/文档/Pytorch_ws/robotcv_pipeline/imgpairs/img11.jpg", cv2.IMREAD_COLOR)

orb = cv2.ORB_create()
kp0, des0 = orb.detectAndCompute(img0, None)
kp1, des1 = orb.detectAndCompute(img1, None)
# cv2.drawKeypoints(img0, kp1, img0)
# cv2.drawKeypoints(img1, kp1, img1)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des0, des1)
topnum = int(len(matches)/7)
finalmatch = sorted(matches, key=lambda mat: mat.distance)[
    0:29]
final_img = cv2.drawMatches(img0, kp0, img1, kp1, finalmatch, None)
cv2.imshow("final", final_img)


# pts0 = np.full((finalmatch.__len__(), 2), [0, 0])
# pts1 = np.full((finalmatch.__len__(), 2), [0, 0])

# for idx in range(finalmatch.__len__()):
#     p0 = kp0[finalmatch[idx].queryIdx].pt
#     p1 = kp1[finalmatch[idx].trainIdx].pt
#     pts0[idx, 0] += p0[0]
#     pts0[idx, 1] += p0[1]
#     pts1[idx, 0] += p1[0]
#     pts1[idx, 1] += p1[1]

pts0 = []
pts1 = []
newkp0 = []
newkp1 = []
for idx in range(finalmatch.__len__()):
    p0 = kp0[finalmatch[idx].queryIdx].pt
    p1 = kp1[finalmatch[idx].trainIdx].pt
    pts0.append(p0)
    newkp0.append(kp0[finalmatch[idx].queryIdx])
    pts1.append(p1)
    newkp1.append(kp1[finalmatch[idx].trainIdx])

i = 28
draw0 = cv2.drawKeypoints(img0, [newkp0[i]], None)
cv2.imshow("draw0", draw0)
draw1 = cv2.drawKeypoints(img1, [newkp1[i]], None)
cv2.imshow("draw1", draw1)
cv2.waitKey(0)

IntM = np.array([[486.9230,         0,         0],
                 [0,  489.3566,         0],
                 [342.5736,  193.4255,    1.0000]])

IM = IntM.T

print("Num= ", len(pts0))
FM = cv2.findFundamentalMat(points1=np.array(pts0), points2=np.array(pts1), method=cv2.FM_RANSAC, ransacReprojThreshold=3)
EM = cv2.findEssentialMat(points1=np.array(pts0), points2=np.array(
    pts1), cameraMatrix=IM, method=cv2.FM_RANSAC, prob=0.98, threshold=3)

EM1 = IM.T.dot(FM[0].dot(IM))

# retval, R, t, mask = cv2.recoverPose(EM[0], np.array(pts0), np.array(pts1), IntM)
# retval, R, t, mask = cv2.recoverPose(
#     EM[0], np.array(pts0), np.array(pts1), IM)
R1, R2, t = cv2.decomposeEssentialMat(EM[0])
retval = cv2.recoverPose(E=EM[0], points1=np.array(pts0), points2=np.array(pts1), cameraMatrix=IM)
R0 = retval[1]
print("R=", R1)
print("t=", t)

ori = np.array([0, 0, 0, 0, 0, 0])
dir = np.array([0, 0, 0.5])
after = R1.dot(dir)
wrong = R2.dot(dir)
other = R0.dot(dir)
final_t = t

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, 0, 0, 0.5)
ax.quiver(0, 0, 0, 10, 0, 0)
ax.quiver(0, 0, 0, 0, 10, 0)
ax.quiver(final_t[0], final_t[1], final_t[2],
          after[0], after[1], after[2], color="green")
ax.quiver(final_t[0], final_t[1], final_t[2],
          wrong[0], wrong[1], wrong[2], color="red")
ax.quiver(0, 0, 0, other[0], other[1], other[2], color="blue")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
plt.show()
cv2.waitKey(0)
