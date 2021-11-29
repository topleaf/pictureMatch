"""
utility to try out different blur level and threshold level
in order to get a good contour rectangle
"""
import cv2
import numpy as np
from edgeDetect import getRequiredContours,warpImg

def func(x):
    pass


cameraResW = 800
cameraResH = 600
scale = 2
wP = 300*scale
hP = 300*scale


if __name__ == '__main__':
    webCam = False
    if webCam:
        camera = cv2.VideoCapture(0)
        camera.set(3,cameraResW)
        camera.set(4,cameraResH)

    else:
        img = cv2.imread("origin.jpg")
    windowName = 'Contour image'
    cv2.namedWindow(windowName)

    tbCannyThr1 = 'canny threshold 1'
    tbCannyThr2 = 'canny threshold 2'
    cv2.createTrackbar(tbCannyThr1, windowName, 0, 255, func)
    cv2.createTrackbar(tbCannyThr2, windowName, 0, 255, func)
    tbBlurlevel = 'blur level'
    cv2.createTrackbar(tbBlurlevel, windowName, 1, 255, func)

    tbMinArea = 'minArea'
    tbMaxArea = 'maxArea'
    cv2.createTrackbar(tbMinArea, windowName, 20, 50000, func)
    cv2.createTrackbar(tbMaxArea, windowName, 20, 50000, func)

    switch = '0 : Origin\n1 : Gray\n2 : blur\n 3: Canny\n 4: dilate\n 5:erode\n 6:Contour\n'
    cv2.createTrackbar(switch, windowName, 0, 6, func)
    cv2.setTrackbarPos(switch, windowName, 0)
    cv2.setTrackbarPos(tbBlurlevel, windowName, 4)
    cv2.setTrackbarPos(tbMinArea, windowName, 2500)
    cv2.setTrackbarPos(tbMaxArea, windowName, 10000)
    cv2.setTrackbarPos(tbCannyThr1, windowName, 80)
    cv2.setTrackbarPos(tbCannyThr2, windowName, 25)
    kernel = np.ones((5, 5))
    while(1):
        if webCam:
            success, img = camera.read()
            if not success:
                break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        threshold_1 = cv2.getTrackbarPos(tbCannyThr1, windowName)
        threshold_2 = cv2.getTrackbarPos(tbCannyThr2, windowName)
        blurr_level = cv2.getTrackbarPos(tbBlurlevel, windowName)
        blurr_level += 1    # avoid 0

        minArea = cv2.getTrackbarPos(tbMinArea, windowName)
        maxArea = cv2.getTrackbarPos(tbMaxArea, windowName)
        s = cv2.getTrackbarPos(switch, windowName)

        if s == 0:
            cv2.imshow(windowName, img)
        elif s == 1:
            cv2.imshow(windowName, gray)
        elif s == 2:
            blur = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 1)
            cv2.imshow(windowName, blur)
        elif s == 3:
            blur = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 1)
            # retval, thresh = cv2.threshold(blur, threshold_1, 255, cv2.THRESH_BINARY)#|cv2.THRESH_OTSU)
            imgCanny = cv2.Canny(blur, threshold_1, threshold_2)
            cv2.imshow (windowName, imgCanny)

        elif s == 4:
            blur = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 1)
            imgCanny = cv2.Canny(blur, threshold_1, threshold_2)
            imgDilate = cv2.dilate(imgCanny, kernel=kernel, iterations=3)
            cv2.imshow (windowName, imgDilate)
        elif s == 5:
            blur = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 1)
            imgCanny = cv2.Canny(blur, threshold_1,threshold_2)
            # imgErode = cv2.erode(imgCanny, kernel,iterations=1)
            # imgDilate = cv2.dilate(imgErode, kernel=kernel, iterations=1)
            # cv2.imshow (windowName, imgDilate)
            imgDilate = cv2.dilate(imgCanny, kernel=kernel, iterations=3)
            imgErode = cv2.erode(imgDilate, kernel, iterations=2)
            cv2.imshow (windowName, imgErode)
        else:
            blur = cv2.blur(gray, (blurr_level, blurr_level), 1)
            imgCanny = cv2.Canny(blur, threshold_1,threshold_2)
            imgDilate = cv2.dilate(imgCanny, kernel=kernel, iterations=3)
            imgErode = cv2.erode(imgDilate, kernel, iterations=2)

            # cv2.imshow (windowName, imgDilate)

            # imgDilate = cv2.dilate(imgCanny, kernel=kernel, iterations=3)
            # imgErode = cv2.erode(imgDilate, kernel,iterations=2)

            contours, hierarchy = cv2.findContours(imgErode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_img = img.copy()
            # get the lcd Screen part rectangle from original img
            contours_img, conts = getRequiredContours(contours_img, hierarchy, contours, minArea=minArea, maxArea=maxArea, draw=True)
            if len(conts) != 0:
                print('hhhh,conts = {}'.format(conts))
                minAreaRectBox = conts[0][3]
                # project the lcd screen to (wP,hP) size, make a imgWarp for the next step
                imgWarp = warpImg(contours_img, minAreaRectBox, wP, hP)
                cv2.imshow('warped lcd screen img', imgWarp)

                # getRequiredContours from imgWarp, looking for symbols displayed on lcd screen
                contours_img2, conts2 = getRequiredContours(imgWarp, hierarchy, contours, minArea=20, maxArea=minArea-200, draw=True)
                if len(conts2) != 0:
                    for symbol in conts2:
                        cv2.polylines(contours_img2, [symbol[2]], True, (255, 0, 0), 2)  #draw symbol's approx in BLUE
                cv2.imshow('warped lcd screen img', contours_img2)


            # for c in conts:
            cv2.imshow(windowName, contours_img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
