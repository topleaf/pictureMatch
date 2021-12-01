import cv2 as cv
import numpy as np
from cv2 import xfeatures2d


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
    print('boxPoints shape is {}'.format(boxPoints.shape))
    newBox = np.zeros_like(boxPoints)
    boxPoints = boxPoints.reshape((4, 2))
    add = boxPoints.sum(1)  # for each point in boxPoints, make a list of add , which is [x1+y1, x2+y2,x3+y3,x4+y4]
    newBox[0] = boxPoints[np.argmin(add)]
    newBox[3] = boxPoints[np.argmax(add)]
    diff = np.diff(boxPoints, axis = 1)
    newBox[1] = boxPoints[np.argmin(diff)]
    newBox[2] = boxPoints[np.argmax(diff)]
    return newBox


# find contours that  are closed graph with minArea,cornerNumber
def getRequiredContours(img, blurr_level, threshold_1,threshold_2,kernel,
                        minArea=4000, maxArea = 50000,cornerNumber=4,
                        draw=True, needPreProcess=True):
    """

    :param img: original image
    :param blurr_level: kernel size for GaussianBlur
    :param threshold_1: canny threshold 1
    :param threshold_2: canny threshold 2
    :param kernel:  dilate and erode kernel
    :param minArea:  contour has larger area than this minimum area
    :param maxArea:  contour has smaller area than this maximum area
    :param cornerNumber:  contour has corners number that are larger than this
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

    contours, hierarchy = cv.findContours(imgErode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:
        print('hierarchy shape is {}'.format(hierarchy.shape))
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
            print('area ={}, minAreaRect box coordinates = {}'.format(area, box))

            #draw the contour's minAreaRect box in RED
            if draw: cv.drawContours(img, [box], 0, (0, 0, 255), 2)

            finalContours.append((area, approx, box))
            # # calculate center and radius of minimum enclosing circle
            # (x,y),r = cv.minEnclosingCircle(c)
            # # cast to integers
            # center = (int(x),int(y))
            # radius = int(r)
            # # draw minEnclosingCircle in blue
            # cv.circle(img, center, radius, (255, 0, 0), 2)

    # cv.imshow(windowName, img)
    # sort the list by contour's area, so that the larger contours are in the first
    finalContours = sorted(finalContours, key=lambda x: x[0], reverse=True)
    return img, finalContours


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


def isolateROI(img, drawRect = True, save = True, blurr_level = 5, threshold_1=81, threshold_2 = 112,
               kernelSize = 5, minArea = 25000, maxArea = 29000,
               windowName = 'find Contours', wP = 600, hP = 600):
    """
    display windows on screen showing images
    :param originalImg:
    :param draw: draw rectangle around detected object or not
    :param save: save to disk or not
    :param wP: width of project size in pixel
    :param hP: height of project size in pixel
    :return: true, the warped image of ROI, projected to (wP,hP) size coodination
            or False, None  if under this situation, no contours has been found as ROI
    """

    if img is None:
        print('img is None')
        raise ValueError
    # print('the original picture shape is {}'.format(img.shape))

    kernel = np.ones((kernelSize, kernelSize))
    contours_img = img.copy()
    contours_img, conts = getRequiredContours(contours_img, blurr_level, threshold_1, threshold_2,
                                              kernel,
                                              minArea=minArea, maxArea=maxArea,
                                              cornerNumber=4, draw=drawRect)
    # print('found satisfactory contours: {}'.format(conts))

    # cv.drawContours(img, contours, -1, (0,255,0), 2)
    # if draw:
    #     cv.imshow('contours', contours_img)
    #     cv.moveWindow('contours', 10, 400)
    cv.imshow(windowName, contours_img)
    if save:
        cv.imwrite(windowName+'.png', contours_img)
    if len(conts) != 0:
        minAreaRectBox = conts[0][2]
        # project the lcd screen to (wP,hP) size, make a imgWarp for the next step
        imgWarp = warpImg(contours_img, minAreaRectBox, wP, hP)
        cv.imshow(windowName+' warped_ROI', imgWarp)

        if save:
            cv.imwrite(windowName+'_ROI_found.png', contours_img)
            cv.imwrite(windowName+'_warped_ROI.png', imgWarp)

        return True, imgWarp  # return warped image
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


def matchByORB(img1, img2, maxMatchedCount):
    """
    use ORB to compare 2 images, give True or False verdict
    requires opencv-contrib-python , to install it , run $ pip install opencv-contrib-python
    :param img1:
    :param img2:
    :param minMatchedCount: draw first maxMatchedCount of keypoints between 2 imgs
    :return:  True or False
    """

    imgTargetGray = img1  #cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    imgTemplateGray = img2  #cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(imgTemplateGray, None)
    kp2, des2 = orb.detectAndCompute(imgTargetGray, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv.drawMatches(imgTemplateGray, kp1, imgTargetGray, kp2,  matches[:maxMatchedCount], imgTargetGray, flags=2)
    cv.imshow('matched window', img3)
    # sift = cv.xfeatures2d.Sift_Create()

    return True


def matchByFLANN(targetImg, queryImg, minMatchCount):
    isMatch = False

    # generate both descriptors
    siftFeatureDetector = cv.xfeatures2d.SIFT_create()
    kp1, targetDescriptor = siftFeatureDetector.detectAndCompute(targetImg, None)
    kp2, queryDescriptor = siftFeatureDetector.detectAndCompute(queryImg, None)

    # create FLANN matcher
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv.FlannBasedMatcher(indexParams,searchParams)

    # try knnMatch
    matches = flann.knnMatch(queryDescriptor, targetDescriptor, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) >= minMatchCount:
        print('this is a potential match,matched descriptor number = {}'.format(len(good)))
        return True
    else:
        print('not a match, matched descriptor number is {},less than {}'.format(len(good),minMatchCount))
        return False

cameraResW = 800
cameraResH = 600
scale = 2
wP = 300*scale
hP = 300*scale

if __name__ == "__main__":
    saveImage = False
    drawRect = False
    targetFileName = 'target_detected.png'
    target_img = cv.imread(targetFileName)  # original img to be compared # '1_targetROI.png')#('./4.png')
    camera = cv.VideoCapture(0)
    camera.set(3,cameraResW)
    camera.set(4,cameraResH)
    # originalFileName = 'originLiveCapture.png'

    targetFound = True
    while targetFound:
        targetFound, imgWarp = isolateROI(target_img, drawRect=drawRect, save=False, blurr_level = 5, threshold_1 =81,
                                          threshold_2 = 112, kernelSize = 5, minArea = 25000, maxArea = 29000,
                                          windowName='target')
        if not targetFound:
            print('please double check target file {}, make sure parameters are good to detect ROI area inside '.format(targetFileName))
            break
        success, live_img = camera.read()
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
        liveFound, liveImgWarp = isolateROI(live_img, drawRect=drawRect, save=saveImage, blurr_level = 5, threshold_1 =81,
                                            threshold_2 = 112, kernelSize = 5, minArea = 25000, maxArea = 29000,
                                            windowName='live_capture')

        if liveFound:
            result = matchByFLANN(imgWarp,liveImgWarp,minMatchCount = 15)
            print('match result = {} '.format(result))
            markImg = live_img.copy()
            cv.putText(markImg, (lambda x: 'Pass' if x else 'Fail')(result), (100,100),
                   cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
            cv.imshow('live_capture', markImg)
        else:
            markImg = target_img.copy()
            cv.putText(markImg, 'Please adjust camera', (100,100),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            cv.imshow('target', markImg)


        # result = matchByORB(imgWarp, liveImgWarp, maxMatchedCount=500)


        # result = matchByTemplate(imgWarp, liveImgWarp, matchThreshold=0.8, draw=True, save=saveImage)
        # matchByTemplate method does not fit for this scenario, since it requires pixel by pixel exact match ,
        # which is hard in engineering environment, the image captured  by camera are pixel-different even in same environment
        # in 2 shots
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

    cv.destroyAllWindows()

