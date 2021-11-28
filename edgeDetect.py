import cv2 as cv
import numpy as np
from filters import SharpenFilter,Embolden5Filter, Embolden3Filter

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

def reorder(boxPoints):
    """
    reorder points in the order of
      1 , 2
      3,  4
    :param boxPoints:
    :return:
    """
    print('boxPoints shape is {}'.format(boxPoints.shape))
    newBox = np.zeros_like(boxPoints)
    boxPoints = boxPoints.reshape((4,2))
    add = boxPoints.sum(1)
    newBox[0] = boxPoints[np.argmin(add)]
    newBox[3] = boxPoints[np.argmax(add)]
    diff = np.diff(boxPoints, axis = 1)
    newBox[1] = boxPoints[np.argmin(diff)]
    newBox[2] = boxPoints[np.argmax(diff)]
    return newBox

# find contours that  are closed graph with minArea,cornerNumber
def getRequiredContours(img, hierarchy, contours, minArea=40000, cornerNumber=4, draw = True):
    """

    :param img:
    :param hierarchy:
    :param contours:
    :param minArea:  contour has larger area than this minimum area
    :param cornerNumber:  contour has corners larger than this
    :return: list of satisfactory contours (c ,area, approx, boundingbox )
    """
    print('hierarchy shape is {}'.format(hierarchy.shape))
    finalContours = []
    for c in contours:
        area = cv.contourArea(c)
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02*peri, True)

        if area >= minArea and len(approx) >= cornerNumber:
            #find bounding box coordinates, and draw it in Green
            x,y,w,h = cv.boundingRect(c)
            if draw: cv.rectangle(img, (x,y),(x+w,y+h),(0,255,0), 2)

            #find minimum area
            rect = cv.minAreaRect(c)

            # calculate coordinates of the minimum area rectangle
            box = cv.boxPoints(rect)

            # normalize coordinates to integers
            box = np.int0(box)
            print('box coordinates = {}'.format(box))
            # box = reorder(box)
            # print('after reorder: box = {}'.format(box))

            #draw contours in RED
            if draw: cv.drawContours(img, [box], 0, (0, 0, 255), 3)

            finalContours.append((c, area, approx, box))
            # # calculate center and radius of minimum enclosing circle
            # (x,y),r = cv.minEnclosingCircle(c)
            # # cast to integers
            # center = (int(x),int(y))
            # radius = int(r)
            # # draw minEnclosingCircle in blue
            # cv.circle(img, center, radius, (255, 0, 0), 2)

    # sort the list by contour's area, so that the larger contours are in the first
    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
    return finalContours


def isolateROI(img, display = True,save = True):
    """

    :param originalImg:
    :param display:
    :param save:
    :return: top,left, w,h coordinates
    """
    if img is None:
        print('img is None')
        raise ValueError
    print('the original picture shape is {}'.format(img.shape))
    if display:
        cv.imshow('original', img)
        cv.moveWindow('original', 10, 100)

    #color transformation to gray space
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if display:
        cv.imshow('gray', img_gray)
        cv.moveWindow('gray', 400, 100)

    #   using scharr operation
    # gradX = cv.Sobel(img_gray,ddepth=cv.CV_32F,dx=1,dy=0,ksize=-1)
    # gradY = cv.Sobel(img_gray,ddepth=cv.CV_32F,dx=0,dy=1,ksize=-1)
    # gradient = cv.subtract(gradX,gradY)
    # gradient = cv.convertScaleAbs(gradient)


    # apply sharpen filter to increase high frequency noise
    # shapenF = SharpenFilter()
    # shapened = img.copy()
    # shapenF.apply(img, shapened)
    # cv.imshow('shapened', shapened)
    # cv.moveWindow('shapened', 800, 100)

    # apply embolden5 filter
    # ebF = Embolden5Filter()
    # ebed = img.copy()
    # ebF.apply(img, ebed)
    # cv.imshow('embodied5', ebed)
    # cv.moveWindow('embodied5', 400, 100)
    #
    # ebF = Embolden3Filter()
    # ebed3 = img.copy()
    # ebF.apply(img, ebed3)
    # cv.imshow('embodied3', ebed3)
    # cv.moveWindow('embodied3', 800, 100)

    #remove high frequency noise
    # using 16X16 kernel to average blur
    blurred = cv.blur(img_gray, (7, 7))
    if display:
        cv.imshow('blurred', blurred)
        cv.moveWindow('blurred', 800, 100)

    # threshold to binary
    retval, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    print('ret_thresh_val = %d' %retval)


    # retval, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # retval, thresh = cv.threshold(img_gray,160,255,cv.THRESH_BINARY)
    # frameMean, frameThresh = self._convert2bipolar(frameGray)
    if display:
        cv.imshow('thresh', thresh)
        cv.moveWindow('thresh',1200, 100)
    if save:
        cv.imwrite('thresh.png', thresh)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print('contours len = {}'.format(len(contours)))
    contours_img = img.copy()
    conts = getRequiredContours(contours_img, hierarchy, contours)
    print('found satisfactory contours: {}'.format(conts))

    # cv.drawContours(img, contours, -1, (0,255,0), 2)
    if display:
        cv.imshow('contours', contours_img)
        cv.moveWindow('contours', 10, 400)
    if save:
        cv.imwrite('contourDetected.png', contours_img)
        # cv.imwrite('resized_50.png', rescaleFrame(img,50))
        # cv.imwrite('resized_176_144.png',setFrameRes(img,176,144))

    approx = conts[0][2]     # the approx of the largest contour
    # if len(approx) == 4:  # with 4 corners
    #     pass
    # minx, miny, maxx, maxy = conts[0, 0], conts[0,1], 0, 0
    # for x,y in conts:
    #     if x < minx:
    #         minx = x
    #     if x > maxx:
    #         maxx = x
    #     if y < miny:
    #         miny = y
    #     if y > maxy:
    #         maxy = y
    # w = maxx - minx
    # h = maxy - miny
    # cv.rectangle(contours_img, (minx,miny),(minx+w,miny+h),(255,255,255), 2)  #draw WHITE line
    cv.drawContours(contours_img, approx,0,(255,255,255),4)
    if display:
        cv.imshow('isolateROI',contours_img)
    if save:
        cv.imwrite('isolateROI.png',contours_img)
    return conts[0][3]  # return boundingbox

    # # get edge detect using Canny
    # cv.imwrite('./pictures/1-canny.jpg', cv.Canny(img_gray, 175, 10))
    # cv.imshow('canny',cv.imread('./pictures/1-canny.jpg'))
    # cv.moveWindow('canny', 800, 100)
    # lcdScreen = img[miny:miny+h, minx:minx+w]
    # if save:
    #     cv.imwrite('original_ROI.png', lcdScreen)
    #     print('please check file original_ROI.png')
    # return minx,miny,w,h

def lineDetection(img,minLineLength,maxLineGap):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 120)

    lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    for item in lines:
        # x1,y1,x2,y2 = zip(line)
        cv.line(img,(item[0,0], item[0,1]),(item[0,2], item[0,3]),(0,255,0),2)
    cv.imshow("edges",edges)
    cv.imshow("lines",img )

    pass

if __name__ == "__main__":
    img = cv.imread('1_targetROI.png')#('./4.png')
    isolateROI(img, True, True)
    cv.imshow('hello', img)
    # left, top, w, h = isolateROI(img, False, True)
    # print('left top corner :(%d,%d),width=%d,height=%d' %(left,top,w,h))
    #
    # lcdScreen = img[top:top+h,left:left+w]
    # cv.imshow('lcdScreen',lcdScreen)
    # lineDetection(lcdScreen, 10, 5)
    cv.waitKey()
    cv.destroyAllWindows()

