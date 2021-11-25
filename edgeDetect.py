import cv2 as cv
import numpy as np

def rescaleFrame(frame,percent=75):
    """

    :param frame:
    :param percent: target frame percentage to original
    :return:
    """
    width = int(frame.shape[1]*percent/100)
    height = int(frame.shape[0]*percent/100)
    dim = (width, height)

    return cv.resize(frame,dim, interpolation=cv.INTER_AREA)

def setFrameRes(frame, width, height):
    """

    :param frame:
    :param width:
    :param height:
    :return:
    """
    return cv.resize(frame,(width,height),interpolation=cv.INTER_AREA)

# find inner rectangle as ROI
def findROI(img,hierarchy,contours):
    print('hierarchy shape is {}'.format(hierarchy.shape))
    i = 0

    for c in contours:
        #find bounding box coordinates, and draw it in Green
        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(img, (x,y),(x+w,y+h),(0,255,0), 2)

        #find minimum area
        rect = cv.minAreaRect(c)

        # calculate coordinates of the minimum area rectangle
        box = cv.boxPoints(rect)

        # normalize coordinates to integers
        box = np.int0(box)
        print('box coordinates = {}'.format(box))
        #draw contours in RED
        cv.drawContours(img, [box], 0, (0, 0, 255), 3)
        if i==1:
            boxROI = box
        # # calculate center and radius of minimum enclosing circle
        # (x,y),r = cv.minEnclosingCircle(c)
        # # cast to integers
        # center = (int(x),int(y))
        # radius = int(r)
        # # draw minEnclosingCircle in blue
        # cv.circle(img, center, radius, (255, 0, 0), 2)
        i += 1

    return boxROI


if __name__ == "__main__":
    img = cv.imread('./1.png')
    print('the original picture shape is {}'.format(img.shape))
    cv.imshow('original', img)
    cv.moveWindow('original', 10,100)

    #color transformation to gray space
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', img_gray)
    cv.moveWindow('gray', 400, 100)

    #   using scharr operation
    # gradX = cv.Sobel(img_gray,ddepth=cv.CV_32F,dx=1,dy=0,ksize=-1)
    # gradY = cv.Sobel(img_gray,ddepth=cv.CV_32F,dx=0,dy=1,ksize=-1)
    # gradient = cv.subtract(gradX,gradY)
    # gradient = cv.convertScaleAbs(gradient)

    #remove high frequency noise
    # using 16X16 kernel to average blur
    blurred = cv.blur(img_gray, (16,16))
    cv.imshow('blurred', blurred)
    cv.moveWindow('blurred',800,100)

    # threshold to binary
    retval, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)


    # retval, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # retval, thresh = cv.threshold(img_gray,160,255,cv.THRESH_BINARY)
    # frameMean, frameThresh = self._convert2bipolar(frameGray)
    cv.imshow('thresh', thresh)
    cv.moveWindow('thresh',1200,100)
    print('ret = %d' %retval)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print('contours len = {}'.format(len(contours)))
    roi = findROI(img,hierarchy,contours)
    print('found ROI cords {}'.format(roi))

    # cv.drawContours(img, contours, -1, (0,255,0), 2)
    cv.imshow('contours', img)
    cv.moveWindow('contours', 10, 400)
    cv.imwrite('contourDetected.png', img)
    cv.imwrite('resized_50.png', rescaleFrame(img,50))
    cv.imwrite('resized_176_144.png',setFrameRes(img,176,144))



    # # get edge detect using Canny
    # cv.imwrite('./pictures/1-canny.jpg', cv.Canny(img_gray, 175, 10))
    # cv.imshow('canny',cv.imread('./pictures/1-canny.jpg'))
    # cv.moveWindow('canny', 800, 100)



    cv.waitKey()

    cv.destroyAllWindows()

