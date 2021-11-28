"""
utility to try out different blur level and threshold level
in order to get a good contour rectangle
"""
import cv2
import numpy as np
from edgeDetect import getRequiredContours

def func(x):
    pass


if __name__ == '__main__':
    img = cv2.imread("1_targetROI.png")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Contour image')

    cv2.createTrackbar('canny threshold 1','Contour image', 0, 255, func)
    cv2.createTrackbar('canny threshold 2','Contour image', 0, 255, func)
    cv2.createTrackbar('blur level','Contour image', 1, 255, func)

    switch = '0 : Origin\n1 : Gray\n2 : blur\n 3: Canny\n 4:Contour\n'
    cv2.createTrackbar(switch, 'Contour image', 0, 4, func)
    cv2.setTrackbarPos(switch, 'Contour image', 1)
    kernel = np.ones((5,5))
    while(1):
        threshold_1 = cv2.getTrackbarPos('canny threshold 1','Contour image')
        threshold_2 = cv2.getTrackbarPos('canny threshold 2','Contour image')
        blurr_level = cv2.getTrackbarPos('blur level','Contour image')
        blurr_level += 1 # avoid 0

        s = cv2.getTrackbarPos(switch, 'Contour image')
        if s == 0:
            cv2.imshow('Contour image', img)
        elif s == 1:
            cv2.imshow('Contour image', gray)
        elif s == 2:
            blur = cv2.blur(gray, (blurr_level,blurr_level),1)
            cv2.imshow('Contour image', blur)
        elif s == 3:
            blur = cv2.blur(gray, (blurr_level,blurr_level),1)
            # retval, thresh = cv2.threshold(blur, threshold_1, 255, cv2.THRESH_BINARY)#|cv2.THRESH_OTSU)
            imgDilate = cv2.dilate(blur,kernel=kernel, iterations=3)
            imgErode = cv2.erode(imgDilate,kernel,iterations=2)
            imgCanny = cv2.Canny(imgErode,threshold_1,threshold_2)

            cv2.imshow('Contour image', imgCanny)
        else:
            blur = cv2.blur(gray, (blurr_level,blurr_level),1)
            # blur = cv2.blur(gray, (blurr_level,blurr_level))
            # retval, thresh = cv2.threshold(blur,threshold_1, 255, cv2.THRESH_BINARY)#|cv2.THRESH_OTSU)
            imgDilate = cv2.dilate(blur,kernel=kernel, iterations=3)
            imgErode = cv2.erode(imgDilate,kernel,iterations=2)
            imgCanny = cv2.Canny(imgErode,threshold_1,threshold_2)
            contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_img = img.copy()
            conts = getRequiredContours(contours_img, hierarchy, contours,draw=True)
            # for c in conts:
            cv2.imshow('Contour image', contours_img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
