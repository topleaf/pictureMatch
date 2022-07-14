# image capture and compare with one queryImg using FLANN match
# feature completed on Dec 1, functionally working.
# Dec 2, start to get more training pictures, load them in advance, compare all of them
# with target image to find the most matched one

import cv2 as cv
import numpy as np
# from cv2 import xfeatures2d
from os import walk
import math


def euclideanDist(vector1, vector2):
    """

    :param vector1: tuple(x,y) of vector
    :param vector2:
    :return:  their euclidean distance
    """

    return abs(math.sqrt((vector2[0]-vector1[0])**2 + (vector2[1]-vector1[1])**2))


#from filters import SharpenFilter,Embolden5Filter, Embolden3Filter

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
    # print('boxPoints shape is {}'.format(boxPoints.shape))
    newBox = np.zeros_like(boxPoints)
    boxPoints = boxPoints.reshape((4, 2))
    add = boxPoints.sum(1)  # for each point in boxPoints, make a list of add , which is [x1+y1, x2+y2,x3+y3,x4+y4]
    newBox[0] = boxPoints[np.argmin(add)]
    newBox[3] = boxPoints[np.argmax(add)]
    diff = np.diff(boxPoints, axis = 1)
    newBox[1] = boxPoints[np.argmin(diff)]
    newBox[2] = boxPoints[np.argmax(diff)]
    return newBox


# find 4 corners that are closed graph with minArea,cornerNumber
def getRequiredContoursByHarrisCorner(img, blurr_level, threshold_1,threshold_2,kernel,
                        blockSize=2,ksize=3,k=0.04,
                        draw=True, needPreProcess=True):
    """

    :param img: original image
    :param blurr_level: kernel size for GaussianBlur
    :param threshold_1: canny threshold 1
    :param threshold_2: canny threshold 2
    :param kernel:  dilate and erode kernel
    :param draw:  draw a rectangle around detected contour  or not ?
    :return: img , list of satisfactory contours_related info (area, approx, boundingbox )
    """
    if needPreProcess:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (blurr_level, blurr_level), 1)
        imgCanny = cv.Canny(blur, threshold_1,threshold_2)
        imgDilate = cv.dilate(imgCanny, kernel=kernel, iterations=3)
        imgErode = cv.erode(imgDilate, kernel, iterations=3)
    else:
        imgErode = img

    grayf32 = np.float32(imgErode)
    dst = cv.cornerHarris(grayf32, blockSize=blockSize, ksize=ksize, k=k)

    if dst is None:
        return False, None
    finalContours = []
    img[dst > 0.01*dst.max()] = [0,0,255]

    return img, finalContours

    dst = cv.dilate(dst, None)
    ret, dst = cv.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(grayf32,np.float32(centroids),(5,5),(-1,-1),criteria)
    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    if draw:
        img[res[:,1],res[:,0]]=[0,0,255]
        img[res[:,3],res[:,2]] = [0,255,0]
        # cv.imwrite('subpixel5.png', img)

    #
    # points = reorder((res[:,1], res[:,0]))

    # for c in contours:
    #     area = cv.contourArea(c)
    #     peri = cv.arcLength(c, True)
    #     approx = cv.approxPolyDP(c, 0.01*peri, True)
    #
    #     # reorder approximate points array in the  order of
    #     # 1(topleft),2(topright),3(bottomleft),4(bottomright)
    #
    #     if area <= maxArea and area >= minArea and len(approx) >= cornerNumber:
    #         #find approx bounding box coordinates, and draw it in Green
    #         x,y,w,h = cv.boundingRect(c)
    #         # if draw:
    #         # cv.rectangle(img, (x,y),(x+w,y+h),(0,255,0), 2)
    #         # cv.putText(img,"area={:.1f}".format(area),(x,y+30),
    #         #            cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),2)
    #
    #         # bbox = cv.boundingRect(approx)
    #         #find minimum area of the contour
    #         rect = cv.minAreaRect(approx)
    #         # (x,y),(w,h) ,angle = rect    x,y is the center, w is width, h is height of the round rectangle
    #         # calculate coordinates of the minimum area rectangle
    #         box = cv.boxPoints(rect)
    #
    #         # normalize coordinates to integers
    #         box = np.int0(box)
    #         # print('area ={}, minAreaRect box coordinates = {}'.format(area, box))
    #
    #         #draw the contour's minAreaRect box in RED
    #         if draw: cv.drawContours(img, [box], 0, (0, 0, 255), 2)
    #
    #         finalContours.append((area, approx, box))
    #         # # calculate center and radius of minimum enclosing circle
    #         # (x,y),r = cv.minEnclosingCircle(c)
    #         # # cast to integers
    #         # center = (int(x),int(y))
    #         # radius = int(r)
    #         # # draw minEnclosingCircle in blue
    #         # cv.circle(img, center, radius, (255, 0, 0), 2)
    #
    # # cv.imshow(windowName, img)
    # # sort the list by contour's area, so that the larger contours are in the first
    # finalContours = sorted(finalContours, key=lambda x: x[0], reverse=True)
    # finalContours.append((200, approx, box))
    return img, finalContours


# find contours that  are closed graph with minArea,cornerNumber,
# used to look for the upper water level inside test tube
# this function plays extremely  important role.!!! be careful of input parameters
def getRequiredContoursByThreshold(img, blurr_level, threshold_level, threshold_type, kernel,
                        minArea=40, maxArea = 500000,cornerNumber=4,
                        draw=1):
    """

    :param img: original image
    :param blurr_level: kernel size for GaussianBlur
    :param threshold_level: user-defined threshold level 0-255
    :param threshold_type: either cv.THRESH_BINARY_INV, cv.THRESH_BINARY or cv2.THRESH_TOZERO
    :param kernel:   erode and dilate kernel, the larger the value, the more effect to remove noise
    :param minArea:  contour has larger area than this minimum area
    :param maxArea:  contour has smaller area than this maximum area
    :param cornerNumber:  contour has corners number that are larger than this
    :param draw:  draw a rectangle and a circle around detected contour  in what color [1,2]
    :return: thresh  img , list of satisfactory contours_related info
            (integer of its boundingbox, center, radius, approx, perimeter ), updated original image
    """
    # step 1, convert original image to gray,
    # blur it and threshold it with user-defined blur_level and thresh_level

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (blurr_level, blurr_level), sigmaX=1, sigmaY=1)
    ret, thresh_img = cv.threshold(blur, threshold_level, 255, threshold_type)  #cv2.THRESH_TOZERO) #

    # step 2:
    # apply close operation to thresh_img
    # 先膨胀再腐蚀。被用来填充前景物体中的小洞，或者前景物体上面的小黑点。使得前景变成更加连续的实体
    # IMPORTANT:
    # this operation can improve consistency of recognition algorithm,
    # because 1. it helps to connect closely-adjacent shapes to one shape, so it reduces contours number
    # 2, in later steps of  _compare, area calculation workload is reduced
    # the larger the kernel size, adjacent shapes will be more likely to form into one shape
    closed = cv.morphologyEx(thresh_img, cv.MORPH_CLOSE, kernel)

    # step 3: find the upper rectangle that contains black contents inside the test tube, its upper boundary will
    # be water level
    contours, hierarchy = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:
        # print('hierarchy shape is {}'.format(hierarchy.shape))
        pass
    finalContours = []
    for c in contours:
        area = cv.contourArea(c)
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.01*peri, True)


        if area <= maxArea and area >= minArea and len(approx) >= cornerNumber:
            #find approx bounding box coordinates, and draw it in Green
            # x,y,w,h = cv.boundingRect(c)
            #find minimum area of the contour
            rect = cv.minAreaRect(approx)
            # (x,y),(w,h) ,angle = rect    x,y is the center, w is width, h is height of the round rectangle
            # calculate coordinates of the minimum area rectangle
            box = cv.boxPoints(rect)

            # normalize coordinates to integers
            box = np.int0(box)


            #  calculate center and radius of minimum enclosing circle,
            # redundent code for this application, no use so far.
            (x, y), r = cv.minEnclosingCircle(c)
            # cast to integers
            center = (int(x), int(y))
            radius = int(r)

            if draw:
                cv.drawContours(img, [box], 0, (0, 0, 255), 20) #draw the contour's minAreaRect box in RED
                #cv.circle(img, center, radius, (255, 0, 0), 2) #  draw minEnclosingCircle in blue
                # reorder approximate points array in the  order of
                # 1(upperleft),2(upperright),3(bottomleft),4(bottomright)
                box = reorder(box)
                # box[0], box[1], box[2], box[3] are (x,y) coordination of
                # upperleft,upperright,bottomleft, bottomright points
                print('water level coordinates = ({},{})---({},{})'.format(box[0][0], box[0][1], box[1][0],box[1][1]))
                cv.putText(img,"({},{})".format(box[0][0], box[0][1]),(box[0][0], box[0][1]),
                           cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 6)  # show its upper-left coordination
                cv.putText(img,"({},{})".format(box[1][0], box[1][1]),(box[1][0], box[1][1]),
                           cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 6)  # show its upper-right coordination

                cv.drawContours(closed, [box], 0, (255, 255, 255), 20) #draw the contour's minAreaRect box in white
                cv.putText(closed,"({},{})".format(box[0][0], box[0][1]),(box[0][0], box[0][1]),
                           cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 6)  # show its upper-left coordination
                cv.putText(closed,"({},{})".format(box[1][0], box[1][1]),(box[1][0], box[1][1]),
                           cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 6)  # show its upper-right coordination

                cv.circle(closed, center, radius, (255, 255, 255), 20) #  draw minEnclosingCircle in white
            finalContours.append((box, center, radius, approx, int(peri)))

    # sort the list by contour's perimeter, so that the larger contours are put at the first
    # finalContours = sorted(finalContours, key=lambda x: x[0], reverse=True)

    # sort the list by contour's center position coordinations(x,y), so that the lower center (x,y) contours
    # are put in the first, which most likely contains the upper water level block, so we only use y coordination.
    finalContours = sorted(finalContours, key=lambda x: x[1][1], reverse=False)
    return closed, finalContours, img




# find contours that  are closed graph with minArea,cornerNumber
# this function plays extremely  important role.!!! be careful of input parameters
def getRequiredContours(img, blurr_level, threshold_1,threshold_2,erodeIter,dilateIter,kernel,interestMask,
                        minArea=4000, maxArea = 50000,cornerNumber=4,
                        draw=1, returnErodeImage =True):
    """

    :param img: original image
    :param blurr_level: kernel size for GaussianBlur
    :param threshold_1: canny threshold 1
    :param threshold_2: canny threshold 2
    :param kernel:   erode and dilate kernel, the larger the value, the more effect to remove noise
    :param minArea:  contour has larger area than this minimum area
    :param maxArea:  contour has smaller area than this maximum area
    :param cornerNumber:  contour has corners number that are larger than this
    :param draw:  draw a rectangle and a circle around detected contour  in what color [1,2]
    :param returnErodeImage: True or false, obsolete
    :return: thresh  img , list of satisfactory contours_related info
            (integer of its perimeter, center, radius, approx, boundingbox ), original image
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (blurr_level, blurr_level), sigmaX=1,sigmaY=1)

    # use interestedMask to fetch only interested area data
    if interestMask.shape == blur.shape:
        blur = np.where(interestMask == 0, 0, blur)
    # ret, thresh_img = cv.threshold(blur, threshLevel, 255, cv.THRESH_BINARY_INV) #cv.THRESH_TOZERO)
    #  CRITICAL: use adaptiveThreshold instead, which is tolerable of camera noise. some frames captured by camera changes
    # in R,G,B very frequently
    # https://www.pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/
    # blockSize, C are empirical values for my application
    # use cv.ADAPTIVE_THRESH_MEAN_C instead of cv.ADAPTIVE_THRESH_GAUSSIAN_C, the latter will degrade quality
    # of those shape that locate close to the edge of the image
    thresh_img = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,
                                      blockSize=21, C=10, dst=None)
    # use interestedMask to fetch only interested area data
    if interestMask.shape == thresh_img.shape:
        thresh_img = np.where(interestMask == 0, 0, thresh_img)

    # apply close operation to thresh_img
    # 先膨胀再腐蚀。被用来填充前景物体中的小洞，或者前景物体上面的小黑点。使得前景变成更加连续的实体
    # IMPORTANT:
    # this operation can improve consistency of recognition algorithm,
    # because 1. it helps to connect closely-adjacent shapes to one shape, so it reduces contours number
    # 2, in later steps of  _compare, area calculation workload is reduced
    # the larger the kernel size, adjacent shapes will be more likely to form into one shape
    closed = cv.morphologyEx(thresh_img, cv.MORPH_CLOSE, kernel)

    # the next 3 lines are redundant, legacy code, no use so far
    imgCanny = cv.Canny(blur, threshold_1, threshold_2)
    imgDilate = cv.dilate(imgCanny, kernel=kernel, iterations=dilateIter)
    imgErode = cv.erode(imgDilate, kernel, iterations=erodeIter)

    contours, hierarchy = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:
        # print('hierarchy shape is {}'.format(hierarchy.shape))
        pass
    finalContours = []
    for c in contours:
        area = cv.contourArea(c)
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.01*peri, True)
        # reorder approximate points array in the  order of
        # 1(topleft),2(topright),3(bottomleft),4(bottomright)

        if area <= maxArea and area >= minArea and len(approx) >= cornerNumber:
            #find approx bounding box coordinates, and draw it in Green
            x,y,w,h = cv.boundingRect(c)
            # if draw:
                # cv.rectangle(img, (x,y),(x+w,y+h),(0,255,0), 2)
                # cv.putText(img,"area={:.1f}".format(area),(x,y+30),
                #            cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),2)

            # bbox = cv.boundingRect(approx)
            #find minimum area of the contour
            rect = cv.minAreaRect(approx)
            # (x,y),(w,h) ,angle = rect    x,y is the center, w is width, h is height of the round rectangle
            # calculate coordinates of the minimum area rectangle
            box = cv.boxPoints(rect)

            # normalize coordinates to integers
            box = np.int0(box)
            # print('area ={}, minAreaRect box coordinates = {}'.format(area, box))

            # # calculate center and radius of minimum enclosing circle
            (x, y), r = cv.minEnclosingCircle(c)
            # cast to integers
            center = (int(x), int(y))
            radius = int(r)

            if draw:
                cv.drawContours(img, [box], 0, (0, 0, 255), 2) #draw the contour's minAreaRect box in RED
                cv.circle(img, center, radius, (255, 0, 0), 2) #  draw minEnclosingCircle in blue
                if not returnErodeImage:   # return thresh_img
                    cv.drawContours(closed, [box], 0, (255, 255, 255), 2) #draw the contour's minAreaRect box in white
                    cv.circle(closed, center, radius, (255, 255, 255), 2) #  draw minEnclosingCircle in white
            finalContours.append((int(peri), center, radius, approx, box))
    # # no need to draw for draw==1, which has been better handled by displayWindowclass
    # if draw == 2 and not returnErodeImage:  # always draw interestMask rectangle in WHITE on thresh_img only,
    #     maskContours, hierarchy = cv.findContours(interestMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #     cv.drawContours(thresh_img, maskContours, -1, color=(255,255,255),thickness=2)

    # sort the list by contour's area, so that the larger contours are in the first
    # finalContours = sorted(finalContours, key=lambda x: x[0], reverse=True)

    # sort the list by contour's center position coordinations(x,y), so that the lower center (x,y) contours
    # are put in the first, this is coarse, only use y coordination.
    finalContours = sorted(finalContours, key=lambda x: x[1][1], reverse=True)
    if returnErodeImage:
        return imgErode, finalContours, img
    else:
        return closed, finalContours, img


def warpImg(img, points, w, h):
    """
    mapping source img with  4 corners points in the order of 1(topleft),2(topright),3(bottomleft),4(bottomright)
    to new imgWarp with 4 corners points at coordinates (0,0),(w,0),(0,h),(w,h)
    :param img:
    :param points:
    :param w:  width in pixel
    :param h: height in pixel
    :return:
    """
    #print(points)
    points = reorder(points)

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)

    imgWarp = cv.warpPerspective(img, matrix, (w, h))
    return imgWarp


def extractValidROI(rawImg, drawRect=False, save=False, wP=600, hP=600, winName='winName',display=False ):
    """
         both training process and predict process must call this to method with exact same parameters
    :param rawImg:
    :param drawRect:
    :param save:
    :param wP:
    :param hP:
    :param winName:
    :param display:
    :return:
    """
    # this method contains tunable parameters that must be consistent between training and predict ,
    # they are blurr_level, threshold_1,threshold_2,kernelSize,minArea,maxArea,wP and hP
    retval, warpImg = isolateROI(rawImg, drawRect=drawRect, save=save, blurr_level=1,
                                 threshold_1=87, threshold_2=132,
                                 kernelSize=5, minArea = 50000, maxArea = 116230,
                                 windowName=winName, wP=wP,
                                 hP=hP, display=display)
    return retval, warpImg

def isolateROI(img, drawRect , save , blurr_level , threshold_1,
               threshold_2 , kernelSize , minArea, maxArea ,
               windowName , wP, hP , display):
    """
    display windows on screen showing images
    :param img: original raw image
    :param drawRect: draw rectangle around detected object or not
    :param save: save to disk or not
    :param wP: width of project size in pixel
    :param hP: height of project size in pixel
    :param display: show img on screen or not ?
    :return: true, the warped image of ROI, projected to (wP,hP) size coodination
            or False, None  if under this situation, no contours has been found as ROI
    """

    if img is None:
        print('img is None')
        return False, None
        raise ValueError
    # print('the original picture shape is {}'.format(img.shape))

    kernel = np.ones((kernelSize, kernelSize))
    contours_img = img.copy()
    blurred_img, conts , contours_img = getRequiredContours(contours_img, blurr_level, threshold_1, threshold_2,2,3,
                                              kernel,
                                              minArea=minArea, maxArea=maxArea,
                                              cornerNumber=4, draw=drawRect,
                                              returnErodeImage=False)
    # do NOT look for  contour within interested area, and warp
    # because edgeDetection under live camera is not stable.
    return True, contours_img  #blurred_img
    # print('found satisfactory contours: {}'.format(conts))

    # cv.drawContours(img, contours, -1, (0,255,0), 2)
    # if draw:
    #     cv.imshow('contours', contours_img)
    #     cv.moveWindow('contours', 10, 400)
    if display: cv.imshow(windowName, contours_img)
    if save:
        cv.imwrite(windowName+'.png', contours_img)
    if len(conts) != 0:
        minAreaRectBox = conts[0][2]
        # project the lcd screen after blur to (wP,hP) size, make a imgWarp for the next step
        imgWarp = warpImg(blurred_img, minAreaRectBox, wP, hP)
        if display: cv.imshow(windowName+' warped_ROI', imgWarp)

        if save:
            cv.imwrite(windowName+'_ROI_found.png', contours_img)
            cv.imwrite(windowName+'_warped_ROI.png', imgWarp)

        return True, imgWarp  # return blurred warped image
    else:
        return False, None      #  not found


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

def matchByTemplate(targetImg, templateImg, matchThreshold=0.9,draw = True,save = False):
    targetH, targetW, _ = targetImg.shape[::]
    templateH, templateW, _ = templateImg.shape[::]

    targetGray = cv.cvtColor(targetImg, cv.COLOR_BGR2GRAY)
    templateGray = cv.cvtColor(templateImg, cv.COLOR_BGR2GRAY)
    # m = np.median(targetGray)
    # n = np.median(templateGray)
    m = n = 20
    print('m={:.2f},n = {:.2f}'.format(m,n))
    _, targetThr = cv.threshold(targetGray, int(m), 255, cv.THRESH_BINARY)
    _, templateThr = cv.threshold(templateGray, int(n), 255, cv.THRESH_BINARY)
    if draw:
        cv.imshow('targetThr', targetThr)
        cv.imshow('templateThr', templateThr)

    result = cv.matchTemplate(targetThr, templateThr, cv.TM_CCOEFF_NORMED)

    minval, maxval, minloc, maxloc = cv.minMaxLoc(result)

    yloc, xloc = np.where(result >= matchThreshold)
    print('maxval = {},length of xloc is {}'.format(maxval, len(xloc)))
    rectangles = []
    for (x, y) in zip(xloc, yloc):
        rectangles.append((int(x), int(y), targetW, targetH))
        rectangles.append((int(x), int(y), targetW, targetH))
    rectangles, weights = cv.groupRectangles(rectangles, 1, 0.2)
    if draw:
        for (x, y, w, h) in rectangles:
            cv.rectangle(targetImg, (x, y), (x + w-1, y + h-1), (255, 0, 0), 3)
        cv.imshow('matchbyTemplate window', result)
    if save:
        cv.imwrite("matchbyTemplate.png", targetImg)
    if maxval > matchThreshold:
        return True
    else:
        return False
    # self._compareResultList.append(maxval)


def matchByORB(img1, img2, maxMatchedCount, display = True):
    """
    use ORB to compare 2 images, give True or False verdict
    requires opencv-contrib-python , to install it , run $ pip install opencv-contrib-python
    :param img1:
    :param img2:
    :param minMatchedCount: draw first maxMatchedCount of keypoints between 2 imgs
    :return:  None
    """
    assert img1 is not None
    assert img2 is not None
    imgQueryGray = img1  #cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    imgTemplateGray = img2  #cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(imgTemplateGray, None)
    kp2, des2 = orb.detectAndCompute(imgQueryGray, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv.drawMatches(imgTemplateGray, kp1, imgQueryGray, kp2,  matches[:maxMatchedCount], imgQueryGray, flags=2)
    img4 = cv.drawKeypoints(image=imgQueryGray, outImage=imgQueryGray, keypoints=kp2, flags=4,color=(51,163,236))
    img5 = cv.drawKeypoints(image=imgTemplateGray, outImage=imgTemplateGray, keypoints=kp1, flags=4,color=(51,163,236))
    if display:
        cv.imshow('matched window', img3)
        cv.imshow('query keypoints',img4)
        cv.imshow('template keypoints',img5)

def loadTrainImgDescriptors(roiFolderName,featureFolderName,imgSuffix='png',descSuffix='npy'):
    """
    load train Info from specified folders into a dictionary
    # :param folderName: raw image files dirname
    :param roiFolderName: roi training image files dirname
    :param featureFolderName: feature descriptor files dirname
    :param imgSuffix: image format suffix, eg, jpg,png,jpeg
    :param descSuffix: descriptor format suffix, 'npy' saved using np.save()
    :return:  trainInfoDict dictionary, eg trainInfoDict:{
            'filename1':{'roiFileName':roifilename1,'descriptor': desc1},
            'filename2':{'roiFileName':roifilename2,'descriptor': desc2},
            ...
            }

    """
    trainInfoDict={}
    for dirpath, dirname, filenames in walk(roiFolderName):
        for file in filenames:
            if file.split('.')[1] == imgSuffix:
                try:    # get raw image filename
                    fileId = int(file.split('.')[0])
                except ValueError:
                    # this is a ROI image file, create key,items in dictionary
                    key = file.split('.')[0].split('_')[0]
                    try:   # only digit accepted
                        n = int(key)
                    except ValueError:
                        print('ignore non-number training image files')
                    else:
                        trainInfoDict[key] = {}
                        trainInfoDict[key]['roiFileName'] = file
                else: # raw image file, use its filename as key
                    pass

    for dirpath, dirname, filenames in walk(featureFolderName):
        for file in filenames:
            if file.split('.')[1] == descSuffix:
                try:    # get filename
                    fileId = int(file.split('.')[0])
                except ValueError:
                        print('ignore non-number featureDescriptor file:{}'.format(file))
                        print('invalid npy filename:{}'.format(file))
                else: # valid descriptor file, load it to respective key as value
                    key = file.split('.')[0]
                    trainInfoDict[key]['descriptor'] = np.load(featureFolderName + file)

    if len(trainInfoDict) == 0:
        return False, None
    else:
        return True, trainInfoDict


def queryDatabaseByFLANN(trainInfoDict, liveImgWarp, expectedId, display, sy, ey, sx, ex,
                         maxDesDifferencePercentage, minMatchedPercentage):
    """
     compare liveImgWarp descriptor with descriptors in trainImgInfos by the key of expectedId,
     return True and the most matched filename in trainImginfos, or False and None
    :param trainInfoDict:{
            'filename1':{'roiFileName':roifilename1,'descriptor': desc1},
            'filename2':{'roiFileName':roifilename2,'descriptor': desc2},
            ...
            }

    :param liveImgWarp:  live warped image
    :param expectedId : in str, the id of expected image
    :param display: True/False
    :param sy:  start y position in pixel
    :param ey: end y position in pixel
    :param sx: start x position in pixel
    :param ex: end x position in pixel    ,above 4 parameters specified interested ROI to be compared
    :param maxDesDifferencePercentage: maximun difference percentage of 2 images descriptors before judging them as a match
    :param minMatchedPercentage : minMatchedPercentage of 2 images Descriptors before judging them as a match
    :return: True,filename or False,None
    """
    try:
        expectedTrainingRoiImg = cv.imread(trainingPicsFolderName +
                                      queryImgDataBase[expectedId]['roiFileName'])
    except KeyError as e:
        print(e)
        print('above training image is not available, please redo training.')
        return False,None

    matchedName = None
    matchedCount, liveDescriptorNum, trainingDescriptorNum =\
        matchByFLANN(liveImgWarp, expectedTrainingRoiImg, expectedId,
                      display, sy, ey, sx, ex)

    # all keypoints descriptors must be the same and the number of keypoints must be same
    #  too strict
    # if matchedCount == liveDescriptorNum and matchedCount == trainingDescriptorNum:

    minDesNum = min(liveDescriptorNum,trainingDescriptorNum)
    maxDesNum = max(liveDescriptorNum,trainingDescriptorNum)
    descNumDiffPer = (maxDesNum-minDesNum)/minDesNum
    matchedPerc = matchedCount/minDesNum
    print('key={},matchedCount={},liveDesNum={},trainingDesNum={},descNumDiffPer={:.2f},'
          'matchedPerc ={:.2f}'.format(
        expectedId, matchedCount,liveDescriptorNum,trainingDescriptorNum,
        descNumDiffPer,matchedPerc))
    if descNumDiffPer <= maxDesDifferencePercentage and matchedPerc >= minMatchedPercentage:
            matchedName = expectedId
    if matchedName is None:
        return False, None
    else:
        print('sample {} MATCHED!!'.format(matchedName))
        return True, matchedName


def matchByFLANN(targetImg, trainingImg, trainingId, display,sy,ey,sx,ex):
    """
    compare both descriptors to get length of matched descriptors
    :param targetImg: warpedImg to be identified
    :param trainingImg: expectImg in training sets to be compared with
    :param trainingId: training img's unique id in str
    :param display : True/False,
    :param sy,ey: start/end pixel position of interest ROI left
    :param sx,ex: start/end pixel position of interest ROI top
    :return: length of matched descriptors, targetImgDescriptorNum,queryImgDescriptorNum
    """

    # set up a mask to select interested zone only
    interestedMask = np.zeros((trainingImg.shape[0], trainingImg.shape[1]), np.uint8)
    interestedMask[sy:ey, sx:ex] = np.uint8(255)
    cv.imshow('mask',interestedMask)

    # generate both descriptors within interestedMask range
    siftFeatureDetector = cv.xfeatures2d.SIFT_create()
    kp1, targetDescriptor = siftFeatureDetector.detectAndCompute(targetImg, interestedMask,None)
    kp2, trainingDescriptor = siftFeatureDetector.detectAndCompute(trainingImg,interestedMask, None)
    if targetDescriptor is None:
        print('ERROR, targetDescriptor is None, return 0,1,10 ,meaning no match')
        return 0, 1, 10
    if trainingDescriptor is None:
        print('ERROR, trainingDescriptor is None(trainingId={}), return 0,1,10 ,'
              'Please tune parameter and redo training'.format(trainingId))
        # raise ValueError
        return 0, 1, 10


    # create FLANN matcher
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)

    flann = cv.FlannBasedMatcher(indexParams, searchParams)

    # try knnMatch
    matches = flann.knnMatch(targetDescriptor, trainingDescriptor, k=2)
    # prepare an empty mask to draw good matches
    matchesMask = [[0, 0] for i in range(len(matches))]

    #David G. Lowe's ratio test, populate the mask
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            good.append(m)

    if display:
        drawParams = dict(matchColor = (0,255,0), singlePointColor = (255,0,0),
                          matchesMask = matchesMask,
                          flags = 0)
        resultImage = cv.drawMatchesKnn(targetImg, kp1, trainingImg, kp2,
                                        matches, None, **drawParams)
        img4 = cv.drawKeypoints(image=trainingImg, outImage=trainingImg, keypoints=kp2, flags=4,color=(51,163,236))
        img5 = cv.drawKeypoints(image=targetImg, outImage=targetImg, keypoints=kp1, flags=4, color=(51,163,236))
        cv.imshow('training keypoints', img4)
        cv.imshow('liveImg keypoints', img5)
        cv.imshow('comparing with training image(at right)', resultImage)

    return len(good), targetDescriptor.shape[0], trainingDescriptor.shape[0]



cameraResW = 1920
cameraResH = 1080
scale = 2
wP = 300*scale
hP = 300*scale
SY,EY = 416, 960
SX,EX = 768, 1364
MAX_DES_DIFF_PER = 0.34     # allowable maximum descriptors number difference in percentage when comparing 2 images
MIN_MATCHED_PER =0.70       # threshold , minimum matched descriptors number in percentage when comparing 2 images
import argparse, os.path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sample main utility to check lcd display")
    parser.add_argument("--device", dest='deviceId', help='video camera ID [0,1,2,3]',default=0, type=int)
    parser.add_argument("--liveMode", dest='webCamera', help='use camera to capture (1) or simulate (0)', default=1, type=int)
    args = parser.parse_args()
    print(args)

    saveImage = False
    drawRect = False
    liveCaptureWindowName = 'live_capture'
    rawTrainingPicsFolderName = r'./pictures/'
    trainingPicsFolderName = r'./rois/'
    trainingFeaturesFolderName =  r'./features/'
    expectedId = 15  # simulate the id of currently capture image
    playbackId = 1   # simulate playback image id
    reloadNeeded = False  # simulate playback image file reload if file changes
    success, queryImgDataBase = loadTrainImgDescriptors(trainingPicsFolderName,
                                                        trainingFeaturesFolderName)
    if not success:
        print('training set loading failed, please check they are valid and retry!')
        exit(-1)


    camera = cv.VideoCapture(args.deviceId)
    camera.set(3, cameraResW)
    camera.set(4, cameraResH)
    # originalFileName = 'originLiveCapture.png'

    live_img = cv.imread(os.path.join(rawTrainingPicsFolderName, '2.png'))
    monitor = True
    while monitor:
        if args.webCamera == 1:
            success, live_img = camera.read()
        else:
            if reloadNeeded:
                live_img = cv.imread(os.path.join(rawTrainingPicsFolderName, str(playbackId)+'.png'))
                reloadNeeded = False

            if live_img is not None:
                success = True
                liveCaptureWindowName = 'live_capture (playback mode)'


        # the following 2 clause is to try out compare target with 1. exact the same picture or 2. part of the same picture
        # test result shows , under 1, matchTemplate compare maxval is 1. perfect match
        #  under 2, matchTemplate compare maxval is 0.68,  the reason are in this situation,
        # isolateROI returns a different imgWarp, due to getRequiredContours() detect a similar but
        # not 100% exact same contour to select the object, which in turn, generate a different warpImg,
        # since matchTemplate() method requires both imgs to be exactly the same, without rotation or angle difference
        # so it failed to give match verdict.
        # ###################### start of try out #####################
        # live_img = cv.imread(targetFileName)
        # live_img = live_img[30:500, 30:500, :]
        # success = True
        ################ end of try out #################################

        if not success:
            continue
        liveFound, liveImgWarp = extractValidROI(live_img, drawRect=drawRect, save=saveImage, wP=wP, hP=wP,
                                            winName=liveCaptureWindowName,display=False)
        if liveFound:
            # result = matchByTemplate(imgWarp, liveImgWarp, matchThreshold=0.8, draw=True, save=saveImage)
            # matchByTemplate method does not fit for this scenario, since it requires pixel by pixel exact match ,
            # which is hard in engineering environment, the image captured  by camera are pixel-different even in same environment
            # in 2 shots
            result, matchedFileName = queryDatabaseByFLANN(queryImgDataBase, liveImgWarp, str(expectedId),
                                                           display=True,
                                                           sy=SY,ey=EY,sx=SX, ex=EX,
                                                           maxDesDifferencePercentage=MAX_DES_DIFF_PER,
                                                           minMatchedPercentage=MIN_MATCHED_PER)
            if result:
                matchedTrainingRoiImg = cv.imread(trainingPicsFolderName +
                                                  queryImgDataBase[matchedFileName]['roiFileName'])
                # matchByORB(matchedTrainingRoiImg, liveImgWarp,
                #            maxMatchedCount=500, display=True)       # draw matching points between 2,visualize them
                verdict = 'expected={}, matched={} '.format(expectedId, matchedFileName)
                print(verdict)
                cv.putText(matchedTrainingRoiImg,"sample id: " + matchedFileName,(10,500),
                           cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
                # cv.imshow('matched training sample', matchedTrainingRoiImg)
            else:   # not a match in training sets
                verdict = 'expected={}, no match '.format(expectedId)
                # cv.destroyWindow('matched training sample')

            markImg = live_img.copy()
            if result:
                cv.putText(markImg, verdict + ' Pass', (10,500),
                                    cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
            else:
                cv.putText(markImg, verdict + ' Fail', (10,500),
                           cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            cv.imshow(liveCaptureWindowName, markImg)
        else:
            try:
                markImg = live_img.copy()
            except AttributeError as e:
                print('warning: empty raw live_img file specified')
                print(e)
            else:
                cv.putText(markImg, 'Please adjust camera or edge detection para', (10,100),
                           cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv.imshow(liveCaptureWindowName, markImg)

        saveImage = False


        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('a') or k == ord('A'): # save current frames to disk
            saveImage = True
        elif k == ord('d') or k == ord('D'):  # enable draw rectangle around contours
            drawRect = True
        elif k == ord('u') or k == ord('U'):  # disable draw rectangle around contours
            drawRect = False
        elif k in range(ord('0'), ord('9'), 1): # simulate expected id
            expectedId = int(chr(k))
        elif k == ord('n') or k == ord('N'):  # simulate load next playback id image
            playbackId += 1
            if playbackId == len(queryImgDataBase):
                playbackId = 0
            reloadNeeded = True
        elif k == ord('b') or k == ord('B'):  # simulate load previous playback id image
            playbackId -= 1
            if playbackId < 0:
                playbackId = len(queryImgDataBase)-1
            reloadNeeded = True
    cv.destroyAllWindows()
    if args.webCamera == 1:
        camera.release()

