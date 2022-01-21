import numpy as np
import cv2
import time

from numba import cuda

DEBUG = False


@cuda.jit
def filter(img, res, mean, hl, width):
    '''CUDA accelerate attemp, not used'''
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    i = int(idx/width)
    j = idx % width
    norm = (img[i, j, 0]**2+img[i, j, 1]**2+img[i, j, 2]**2)**0.5
    angle = (img[i, j, 0]*mean[0] + img[i, j, 1]
             * mean[1] + img[i, j, 2]*mean[2])/norm
    spot = (img[i, j, 0] > hl[0] and img[i, j, 1]
            > hl[1] and img[i, j, 2] > hl[2])
    res[i, j] = 1-1*(angle > 0.995 or spot)


class BcRd():
    def __init__(self, img) -> None:
        self.img = img
        self.mask = np.ones((len(img), len(img[0])))
        self.crt = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        self.num_mask = 4
        self.hl = np.array([255, 255, 255])*0.96

    def GenMask(self, bbox: list):
        rsz = False
        fgimg = self.img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        rawshape = (fgimg.shape[1], fgimg.shape[0])
        '''if large image: resize, accelerate & reduce noise'''
        if fgimg.size > 1000000:
            rsz = True
            fgimg = cv2.resize(
                fgimg, (int(fgimg.shape[1]/4), int(fgimg.shape[0]/4)))
        # elif fgimg.size > 500000:
        #     rsz = True
        #     fgimg = cv2.resize(
        #         fgimg, (int(fgimg.shape[1]/2), int(fgimg.shape[0]/2)))

        '''bilateral filter: reduce noise'''
        fgimg = cv2.bilateralFilter(fgimg, 5, 100, 10)
        # cv2.imshow("blur", fgimg)
        # cv2.waitKey(0)
        fgdata = np.array(fgimg.reshape(-1, 3), dtype=np.float32)

        '''kmeans: extract major part'''
        ret = cv2.kmeans(fgdata, self.num_mask, None, self.crt,
                         4, cv2.KMEANS_RANDOM_CENTERS)
        kmmask = ret[1].reshape(fgimg.shape[0:2])
        masks = []
        for i in range(self.num_mask):
            masks.append(np.array(kmmask ^ 0 == i, dtype=np.uint64))

        '''calculate the background mean RGB value as background typical vector'''
        self.mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0
        background = self.img * np.expand_dims(self.mask, 2).repeat(3, axis=2)
        imgdata = background.reshape(1, -1, 3)[0]
        self.m = np.mean(imgdata, axis=0)
        self.m /= np.linalg.norm(self.m)

        # filter[2025, 1024](fgimg, fgmask, self.m, self.hl, fgimg.shape[1])

        '''eliminate highlight spots, if R>hl && G>hl && B>hl, set the pixel as background'''
        hlmask = np.full(fgdata.shape, self.hl)
        fgmask2 = hlmask - fgdata  # spot == 0
        fgmask2 = np.sign(fgmask2)
        fgmask2 = fgmask2.dot(np.array([1, 1, 1]))
        fgmask2 = np.array(np.sign(3+fgmask2), dtype=np.uint8)

        fgnorm = np.sqrt(
            np.square(fgdata[:, 0])+np.square(fgdata[:, 1])+np.square(fgdata[:, 2]))
        fgmask1 = fgdata.dot(self.m)
        fgmask1 /= fgnorm
        fgmask1 = (np.sign(0.995-fgmask1)+1)/2
        fgmask = np.array(
            (fgmask1 * fgmask2).reshape(fgimg.shape[0:2]), dtype=np.uint8)

        '''
        the code above can be interpreted as the followings:

        for i in range(len(res)):
            for j in range(len(res[0])):
                if self.img[i+bbox[0], j+bbox[1]].dot(self.m)/(np.linalg.norm(self.m)*np.linalg.norm(self.img[i+bbox[0], j+bbox[1]])) > 0.995 or sum(self.img[i+bbox[0], j+bbox[1]] > self.hl) == 3:
                    fgmask[i, j] = 0
        '''

        if DEBUG:
            cv2.imshow("fgmask", np.array(fgmask*255, dtype=np.uint8))
            cv2.waitKey(0)

        '''
        calculate recall and precision of each kmeans component
        it's hard to choose a criteria to choose components, use recall
        '''
        precision = []
        recall = []
        # part = []
        fgcount = sum(sum(fgmask))
        for i in range(self.num_mask):
            precision.append(sum(sum(masks[i]*fgmask))/sum(sum(masks[i])))
            recall.append(sum(sum(masks[i]*fgmask))/fgcount)
            # part.append(sum(sum(masks[i]))/fgmask.size)
        score = 0.3*np.array(precision) + 0.7*np.array(recall)
        index = np.argsort(recall)[::-1]

        if DEBUG:
            print("precision= ", precision)
            print("recall= ", recall)
            print("srcore= ", score)
            # print("part= ", part)
            print("index= ", index)

        '''choose the best and the second best(if small enough) components as the frontground'''
        maskfinal = masks[index[0]]
        if(recall[index[0]]+recall[index[1]] < 0.7):
            maskfinal += masks[index[1]]
        # cv2.imshow("maskfinal", np.array(maskfinal*255, dtype=np.uint8))
        # cv2.waitKey(0)

        if DEBUG:
            cv2.imshow("mask0", np.array(masks[0]*255, dtype=np.uint8))
            cv2.imshow("mask1", np.array(masks[1]*255, dtype=np.uint8))
            cv2.imshow("mask2", np.array(masks[2]*255, dtype=np.uint8))
            cv2.imshow("mask3", np.array(masks[3]*255, dtype=np.uint8))
            cv2.imshow("kmeans", np.array(
                kmmask*254/self.num_mask, dtype=np.uint8))
            cv2.imshow("maskfinal", np.array(maskfinal*255, dtype=np.uint8))
            cv2.imshow("maskres", np.array(
                maskfinal*fgmask*255, dtype=np.uint8))
            cv2.waitKey(0)

        '''add the kmeans mask to the original mask'''
        rawres = np.array(maskfinal*fgmask*255, dtype=np.uint8)
        if rsz:
            rawres = cv2.resize(rawres, rawshape)

        '''use connect components to eliminate small noisy spots'''
        retval, labels, stats, cents = cv2.connectedComponentsWithStats(rawres)
        if len(stats) > 1:
            maxcomp = stats[1:, 4].max()/5

            for i in range(retval):
                if stats[i, 4] < maxcomp:
                    labels[stats[i, 1]:stats[i, 1]+stats[i, 3], stats[i, 0]:stats[i, 0]+stats[i, 2]] = np.where(
                        labels[stats[i, 1]:stats[i, 1]+stats[i, 3], stats[i, 0]:stats[i, 0]+stats[i, 2]] == i, 0, labels[stats[i, 1]:stats[i, 1]+stats[i, 3], stats[i, 0]:stats[i, 0]+stats[i, 2]])

        res = np.array(np.sign(labels), dtype=np.uint8)

        return res


if __name__ == '__main__':
    # img = cv2.imread(
    #     "/home/tim/文档/Pytorch_ws/fasterrcnn_robot/ImgAnno/Anno/1-2/JPGImages/000860.jpg", cv2.IMREAD_COLOR)
    img = cv2.imread(
        "/home/tim/文档/Pytorch_ws/fasterrcnn_robot/ImgAnno/Anno/1-2/JPGImages/000024.jpg", cv2.IMREAD_COLOR)
    cv2.imshow("ori", img)
    st = time.time()
    seg = BcRd(img)
    # res = seg.GenMask([0,  637,  212, 1854])
    res = seg.GenMask([0,  190,  742, 1152])
    ed = time.time()
    print("time= ", ed-st)
    cv2.imshow("res", res)
    cv2.waitKey(0)
