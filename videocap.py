import cv2

cam = cv2.VideoCapture(4)
size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# vw = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc(
#     'M', 'P', 'E', 'G'), 30, size)

while True:
    ret, frame = cam.read()
    if ret:
        cv2.imshow("cam", frame)
        # vw.write(frame)
        if cv2.waitKey(33) == 115:
            # cv2.imwrite("circle0.jpg", frame)
            break
    else:
        break
cam.release()
# vw.release()
