# import numpy as np
# import cv2 as cv
# filename = '1.png'
# img = cv.imread(filename)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv.cornerHarris(gray,2,3,0.04)
# #result is dilated for marking the corners, not important
# dst = cv.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]
# cv.imshow('dst',img)
# if cv.waitKey(0) & 0xff == 27:
#     cv.destroyAllWindows()
#
# utility to try out different blur level and threshold level
# in order to get a good contour rectangle
#
# press 'a' key to save current images(original,detected,warped_image) to disk for late checking
#     press esc key to exit

import cv2
import numpy as np
from edgeDetect import warpImg,getRequiredContoursByHarrisCorner

def func(x):
    pass


cameraResW = 800
cameraResH = 600
scale = 2
wP = 300*scale
hP = 300*scale


if __name__ == '__main__':
    webCam = True
    if webCam:
        camera = cv2.VideoCapture(0)
        camera.set(3,cameraResW)
        camera.set(4,cameraResH)
        originalFileName = 'originLiveCapture.png'

    else:
        originalFileName = "./pictures/1.png"
        img = cv2.imread(originalFileName)

    windowName = 'looking for main Contour image by corner detection '
    cv2.namedWindow(windowName)

    saveLcdScreenMarked = False
    saveWarpImg = False
    saveLcdInternal = False
    drawRect = False


    tbHarrisBlockSize = 'Harris BlockSize'
    tbHarrisKSize = "Harris KSize"
    tbHarrisK = 'Harris K'
    cv2.createTrackbar(tbHarrisBlockSize, windowName, 2, 100, func)
    cv2.createTrackbar(tbHarrisKSize, windowName, 2, 31, func)
    cv2.createTrackbar(tbHarrisK, windowName, 1, 100, func)

    tbCannyThr1 = 'canny threshold low'
    tbCannyThr2 = 'canny threshold high'
    cv2.createTrackbar(tbCannyThr1, windowName, 0, 255, func)
    cv2.createTrackbar(tbCannyThr2, windowName, 0, 255, func)
    tbBlurlevel = 'blur level'
    cv2.createTrackbar(tbBlurlevel, windowName, 1, 50, func)

    tbMinArea = 'minArea'
    tbMaxArea = 'maxArea'
    cv2.createTrackbar(tbMinArea, windowName, 20, 50000, func)
    cv2.createTrackbar(tbMaxArea, windowName, 20, 500000, func)

    switch = '0 : Origin\n1 : Gray\n2 : blur\n 3: Canny\n 4: dilate\n 5:erode\n'
    cv2.createTrackbar(switch, windowName, 0, 5, func)
    cv2.setTrackbarPos(switch, windowName, 5)
    cv2.setTrackbarPos(tbBlurlevel, windowName, 4)
    cv2.setTrackbarPos(tbMinArea, windowName, 50000)
    cv2.setTrackbarPos(tbMaxArea, windowName, 78773)
    cv2.setTrackbarPos(tbCannyThr1, windowName, 86)
    cv2.setTrackbarPos(tbCannyThr2, windowName, 245)

    cv2.setTrackbarPos(tbHarrisBlockSize, windowName, 2)
    cv2.setTrackbarPos(tbHarrisKSize, windowName, 3)
    cv2.setTrackbarPos(tbHarrisK, windowName, 4)

    kernel = np.ones((5, 5))
    while(1):
        if webCam:
            success, img = camera.read()
            if not success:
                break
        threshold_1 = cv2.getTrackbarPos(tbCannyThr1, windowName)
        threshold_2 = cv2.getTrackbarPos(tbCannyThr2, windowName)
        blurr_level = cv2.getTrackbarPos(tbBlurlevel, windowName)
        blurr_level = (lambda x: x+1 if x % 2 == 0 else x)(blurr_level)    # avoid even

        minArea = cv2.getTrackbarPos(tbMinArea, windowName)
        maxArea = cv2.getTrackbarPos(tbMaxArea, windowName)
        s = cv2.getTrackbarPos(switch, windowName)

        harrisBlkSize = cv2.getTrackbarPos(tbHarrisBlockSize, windowName)
        harrisKSize = cv2.getTrackbarPos(tbHarrisKSize, windowName)
        harrisKSize = (lambda x: x+1 if x % 2 == 0 else x)(harrisKSize)    # avoid even
        harrisK = cv2.getTrackbarPos(tbHarrisK, windowName)*0.01

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if s == 0:
            cv2.imshow(windowName, img)
        elif s == 1:
            cv2.imshow(windowName, gray)
        elif s == 2:
            blur = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 1)
            cv2.imshow(windowName, blur)
        elif s == 3:
            blur = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 1)
            imgCanny = cv2.Canny(blur, threshold_1, threshold_2)
            cv2.imshow (windowName, imgCanny)

        elif s == 4:
            blur = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 1)
            imgCanny = cv2.Canny(blur, threshold_1, threshold_2)
            imgDilate = cv2.dilate(imgCanny, kernel=kernel, iterations=3)
            cv2.imshow (windowName, imgDilate)
        elif s == 5:
            # get the lcd Screen part rectangle from original img
            contours_img = img.copy()
            contours_img, conts = getRequiredContoursByHarrisCorner(contours_img, blurr_level, threshold_1, threshold_2,
                                                      kernel,
                                                      blockSize=harrisBlkSize, ksize=harrisKSize,k=harrisK, draw=drawRect,
                                                      needPreProcess=True)
            cv2.imshow(windowName,contours_img)
            if len(conts) != 0:
                print('conts = {}'.format(conts))
                minAreaRectBox = conts[0][2]
                # project the lcd screen to (wP,hP) size, make a imgWarp for the next step
                imgWarp = warpImg(contours_img, minAreaRectBox, wP, hP)
                cv2.imshow('warped lcd screen img', imgWarp)

            # end of getcontours
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord('a') or k == 'A':
        # save current frames(lcdScreenMarkedout, warpImg and contours_img2) to disk
            saveLcdScreenMarked = True
            saveWarpImg = True
            saveLcdInternal = True
        elif k == ord('d') or k == ord('D'):  # enable draw rectangle around contours
            drawRect = True
        elif k == ord('u') or k == ord('U'):  # disable draw rectangle around contours
            drawRect = False

    cv2.destroyAllWindows()

