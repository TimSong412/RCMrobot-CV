from utils.config import opt
from PIL import Image
from propose import BBProposer
# from circlepropose import Circle
import time
import cv2
import numpy as np
from matting import BcRd
import matplotlib.pyplot as plt
DEBUG = False
ESS3D = False


def GenTrans(**kwargs):
    # opt._parse(kwargs)
    # bbox: [[top, left, down, right]]
    '''
    pre = Image.open(opt.pre_img)
    post = Image.open(opt.post_img)
    preimg = np.asarray(pre, dtype=np.uint8)
    postimg = np.asarray(post, dtype=np.uint8)
    '''
    preimg = cv2.imread(opt.pre_img, cv2.IMREAD_COLOR)
    postimg = cv2.imread(opt.post_img, cv2.IMREAD_COLOR)
    pre = Image.fromarray(preimg)
    post = Image.fromarray(postimg)
    # '''

    Predictor = BBProposer(opt)
    # Predictor = Circle()
    print("start")
    st = time.time()

    bbox1, label1, score1 = Predictor.predict(pre)
    # bbox1 = Predictor.predict(preimg)
    bbox1_0 = np.array(bbox1[0][0], dtype=int)

    # print("----box1----", bbox1_0)

    ed = time.time()
    print("t= ", ed-st)
    bbox2, label2, score2 = Predictor.predict(post)
    # bbox2 = Predictor.predict(postimg)
    bbox2_0 = np.array(bbox2[0][0], dtype=int)

    region1 = preimg[bbox1_0[0]:bbox1_0[2], bbox1_0[1]:bbox1_0[3]]
    region2 = postimg[bbox2_0[0]:bbox2_0[2], bbox2_0[1]:bbox2_0[3]]

    # TODO:
    """
    CUDA accelerate
    over-exposed or highlight(fluid) => reduce, interpolate
    """
    bg1 = BcRd(preimg)
    bg2 = BcRd(postimg)

    # mask: 0/1
    mask1 = bg1.GenMask(bbox1_0)
    mask2 = bg2.GenMask(bbox2_0)
    if DEBUG:
        cv2.imshow("mask1", mask1)
        cv2.imshow("mask2", mask2)
        cv2.waitKey(0)

    if ESS3D:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(region1, mask1)
        kp2, des2 = sift.detectAndCompute(region2, mask2)
        # cv2.drawKeypoints(region1, kp1, region1)
        # cv2.drawKeypoints(region2, kp2, region2)
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        # matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # matches = matcher.knnMatch(des1, des2, k=2)
        # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        # des11 = np.array(des1, dtype=np.float32)
        # des22 = np.array(des2, dtype=np.float32)
        matches = matcher.match(des1, des2)

        matchesMask = [[1, 0] for i in range(len(matches))]
        # for i, (m, n) in enumerate(matches):
        #     if m.distance < 0.9 * n.distance:
        #         matchesMask[i] = [1, 0]
        #     else:
        #         print("ok")

        finalmatch = sorted(matches, key=lambda mat: mat.distance)[
            0:int(len(matches)/5)]
        final_img = cv2.drawMatches(
            region1, kp1, region2, kp2, finalmatch, None)
        cv2.imshow("final", final_img)

        # drawParams = dict(matchColor=(0, 255, 0),
        #                   singlePointColor=(255, 0, 0),
        #                   matchesMask=matchesMask[:50],
        #                   flags=0)
        # resultImage = cv2.drawMatchesKnn(
        #     region1, kp1, region2, kp2, matches[:50], None, **drawParams)
        # cv2.imshow("final", resultImage)

        cv2.waitKey(0)
        # pts1 = np.full((finalmatch.__len__(), 2), [
        #                bbox1[0][0][1], bbox1[0][0][0]])
        # pts2 = np.full((finalmatch.__len__(), 2), [
        #                bbox2[0][0][1], bbox2[0][0][0]])

        # for idx in range(finalmatch.__len__()):
        #     p1 = kp1[finalmatch[idx].queryIdx].pt
        #     p2 = kp2[finalmatch[idx].trainIdx].pt
        #     pts1[idx, 0] += p1[0]
        #     pts1[idx, 1] += p1[1]
        #     pts2[idx, 0] += p2[0]
        #     pts2[idx, 1] += p2[1]

        # np.save("robotcv_pipeline/pts1", pts1)
        # np.save("robotcv_pipeline/pts2", pts2)

    pre_gray = cv2.cvtColor(preimg, cv2.COLOR_RGB2GRAY)
    post_gray = cv2.cvtColor(postimg, cv2.COLOR_RGB2GRAY)
    overlap = np.array(pre_gray*0.3+post_gray*0.7, dtype=np.uint8)
    cv2.imshow("overlap", overlap)
    flow = cv2.calcOpticalFlowFarneback(
        pre_gray, post_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    totalmask1 = np.zeros(pre_gray.shape[0:2], dtype=np.uint8)

    totalmask1[bbox1_0[0]:bbox1_0[2], bbox1_0[1]:bbox1_0[3]] = mask1
    finalimg = np.array(preimg *
                        np.expand_dims(totalmask1, 2).repeat(3, axis=2), dtype=np.uint8)
    cv2.imshow("finalimg", finalimg)

    boarder = np.zeros((preimg.shape[0], preimg.shape[1]), dtype=np.uint8)
    boarder[10:preimg.shape[0]-10, 10:preimg.shape[1]-10] += 1
    flow *= np.expand_dims(totalmask1, 2).repeat(2, axis=2)
    flow *= np.expand_dims(boarder, 2).repeat(2, axis=2)
    flowdata = flow.reshape(1, -1, 2)[0]
    shift = np.mean(flowdata, axis=0)
    print("shift= ", shift)
    # print("flowarea ", flow[800:810, 1180:1200])

    # X, Y = np.mgrid[0: 1080, 0: 1920]
    # U = flow[:, :, 0].reshape(-1)*10
    # V = flow[:, :, 1].reshape(-1)*10
    # C = np.sqrt(U**2+V**2)
    # plt.quiver(Y, X[:: -1], V, -U, C)
    # plt.show()
    plt.quiver(0, 0, shift[1], -shift[0])
    plt.show()
    cv2.waitKey(0)


def Initial():
    pass


if __name__ == '__main__':
    GenTrans()
