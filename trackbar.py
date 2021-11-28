
import cv2
import numpy as np

def func(x):
    pass


if __name__ == '__main__':
    img = cv2.imread("1.png")
    cv2.namedWindow('image')

    cv2.createTrackbar('threshold 1','image', 0, 255, func)
    cv2.createTrackbar('threshold 2','image', 0, 255, func)

    switch = 'Canny 1 : ON\n0 : OFF'
    cv2.createTrackbar(switch, 'image', 0, 1, func)
    cv2.setTrackbarPos(switch,'image',1)
    while(1):
        # cv2.setTrackbarPos('R','image',128)
        threshold_1 = cv2.getTrackbarPos('threshold 1','image')
        threshold_2 = cv2.getTrackbarPos('threshold 2','image')

        # b = cv2.getTrackbarPos('B','image')

        s = cv2.getTrackbarPos(switch,'image')
        if s == 0:
            cv2.imshow('image', img)
        else:
            edges = cv2.Canny(img,threshold_1,threshold_2)
            cv2.imshow('image', edges)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
