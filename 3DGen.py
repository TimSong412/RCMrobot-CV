import numpy as np
import matplotlib.pyplot as plt


IntM = np.array([[486.9230,         0,         0],
                 [0,  489.3566,         0],
                 [342.5736,  193.4255,    1.0000]])

# FM = np.array([[-3.89128298e-07, -1.54689975e-05,  1.74070622e-03],
#                [1.57532857e-05, -2.56721569e-07, -1.44124280e-02],
#                [-2.42496412e-03,  1.36581025e-02,  1.00000000e+00]])

FM = np.array([[4.97073735e-06, -3.64796791e-06, -2.53248169e-04],
               [3.60598387e-06,  5.40936065e-06, -2.74977377e-03],
               [-3.35546621e-03, -1.40128309e-04,  1.00000000e+00]])
a = np.array([211.0, 99.0, 1])
b = np.array([95.04000091552734, 220.32000732421875, 1])


# EM = np.array([[0.19947568, -0.25227902, -0.18209376],
#                [0.54452202,  0.42148242,  0.15491722],
#                [0.39137663, -0.36493368, -0.2808974]])

# EM = np.array([[-0.03926061,  0.43962666, -0.11671813],
#                [-0.51946683, -0.12437333,  0.4356901],
#                [0.24063818, -0.50827314, -0.01323604]])

# EM = IntM.T.dot(FM).dot(IntM)

# print(EM)

# R1, R2, t = cv2.decomposeEssentialMat(EM)


W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

# print(U)
# print(S)
# print(Vt)

# R = np.array([[0.30970206,  0.70521908, - 0.63777009],
#              [-0.62713885,  0.65566847,  0.42047084],
#               [0.7146898,   0.26974972,  0.64533215]])
# t = np.array([[0.85253341],
#               [0.06081107],
#               [-0.5191231]])

# R = np.array([[0.96721383,  0.14223534,  0.21039611],
#              [-0.16322899,  0.98283734,  0.08594803],
#               [-0.19456031, -0.11747287,  0.97383079]])
# t = np.array([[0.76362802],
#               [0.22297815],
#               [0.60593151]])

# R = np.array([[0.64099632,  0.68379447, -0.34863856],
#               [-0.69575103,  0.70945272,  0.11228244],
#               [0.32412068,  0.17059301,  0.93050729]])

# t = np.array([[0.86589566],
#               [-0.35697105],
#               [-0.35042314]])

# R,t with correct IM

# 01
# R = np.array([[0.64099632,  0.68379447, -0.34863856],
#              [-0.69575103,  0.70945272,  0.11228244],
#               [0.32412068,  0.17059301,  0.93050729]])
# t = np.array([[0.86589566],
#              [-0.35697105],
#               [-0.35042314]])

# 23
R = np.array([[0.86276587,  0.06815086,  0.50098954],
             [-0.08551253,  0.99626795,  0.01173831],
              [-0.49831984, -0.0529683,   0.86537373]])
t = np.array([[-0.55003034],
              [0.05956165],
              [-0.83301803]])

ori = np.array([0, 0, 0, 0, 0, 0])
dir = np.array([0, 0, 0.5])
after = R.T.dot(dir)

print(np.linalg.norm(t))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, 0, 0, 0.5)
ax.quiver(-t[0], -t[1], -t[2], after[0], after[1], after[2], color="red")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
plt.show()
