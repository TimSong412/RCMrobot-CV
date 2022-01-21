import cv2
import numpy as np
DEBUG = False


class Circle():
    def __init__(self) -> None:
        pass

    def predict(self, img):
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayimg = cv2.GaussianBlur(grayimg, (5, 5), 15)
        cannyout = cv2.Canny(grayimg, 100, 200)
        contours, hierarchy = cv2.findContours(
            cannyout, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # retval = cv2.fitEllipse(contours[0])
        # outimg = cv2.ellipse(img, retval, (0, 0, 255), thickness=2)
        # sortedcont = sorted(contours, key=lambda ct: len(ct), reverse=True)
        eps = []
        for ct in contours:
            if len(ct) < 15:
                eps.append(1)
            else:
                ret = cv2.fitEllipse(ct)
                eps.append(abs(1-ret[1][0]/ret[1][1]))

        idx = np.argsort(eps)

        if DEBUG:
            # cv2.drawContours(img, sortedcont, 2, (255, 0, 0), 2)
            # cv2.drawContours(img, sortedcont, 3, (0, 0, 255), 2)
            for ct in contours:
                print(len(ct))
                ret = cv2.fitEllipse(ct)
                e = ret[1][0]/ret[1][1]
                print("e= ", e)

            cv2.imshow("contours", img)

            cv2.waitKey(0)

        box = cv2.boundingRect(contours[idx[0]])
        [x, y, w, h] = box
        if DEBUG:
            cv2.imshow("blur", grayimg)
            newimg = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("box", newimg)
            cv2.waitKey(0)
        return [[[max(y-20, 0), max(x-20, 0), min(y+h+20, img.shape[0]), min(x+w+20, img.shape[1])]]]


if __name__ == "__main__":
    img = cv2.imread(
        "/home/tim/文档/Pytorch_ws/robotcv_pipeline/circle0.jpg", cv2.IMREAD_COLOR)
    proposer = Circle()
    box = proposer.predict(img)
    [t, l, d, r] = box[0][0]
    cv2.rectangle(img, (l, t), (r, d), (0, 255, 0), 2)
    cv2.imshow("box", img)
    cv2.waitKey(0)
