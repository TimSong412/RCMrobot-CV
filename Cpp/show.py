import numpy as np
import cv2
import matplotlib.pyplot as plt


fs = cv2.FileStorage(
    "/home/tim/文档/Pytorch_ws/robotcv_pipeline/Cpp/transform.xml", cv2.FileStorage_READ)
R1 = np.array(fs.getNode("R1").mat(), dtype=np.float32)
R2 = np.array(fs.getNode("R2").mat(), dtype=np.float32)
t = np.array(fs.getNode("t").mat(), dtype=np.float32)
R0 = np.array(fs.getNode("R0").mat(), dtype=np.float32)
t0 = np.array(fs.getNode("t0").mat(), dtype=np.float32)
fs.release()

ori = np.array([0, 0, 0, 0, 0, 0])
dir = np.array([0, 0, 0.5])
after = R1.T.dot(dir)
wrong = R2.T.dot(dir)
other = R0.T.dot(dir)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, 0, 0, 0.5)
ax.quiver(0, 0, 0, 10, 0, 0)
ax.quiver(0, 0, 0, 0, 10, 0)
ax.quiver(t[0],  t[1],  t[2],
          after[0], after[1], after[2], color="green")
ax.quiver(t[0],  t[1],  t[2],
          wrong[0], wrong[1], wrong[2], color="red")
ax.quiver(t0[0], t0[1], t0[2], other[0], other[1], other[2], color="blue")

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
plt.show()
