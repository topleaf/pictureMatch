# image capture and compare with one queryImg using FLANN match
# feature completed on Dec 1, functionally working.
# Dec 2, start to get more training pictures, load them in advance, compare all of them
# with target image to find the most matched one

import cv2 as cv
import numpy as np
from cv2 import xfeatures2d
from os import walk


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
        return False,None

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
    points = reorder((res[:,1], res[:,0]))
    finalContours = []
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


def isolateROI(img, drawRect = True, save = True, blurr_level = 5, threshold_1=81,
               threshold_2 = 112, kernelSize = 5, minArea = 25000, maxArea = 29000,
               windowName = 'find Contours', wP = 600, hP = 600, display=True):
    """
    display windows on screen showing images
    :param originalImg:
    :param draw: draw rectangle around detected object or not
    :param save: save to disk or not
    :param wP: width of project size in pixel
    :param hP: height of project size in pixel
    :param display: show img on screen or not ?
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
    if display:cv.imshow(windowName, contours_img)
    if save:
        cv.imwrite(windowName+'.png', contours_img)
    if len(conts) != 0:
        minAreaRectBox = conts[0][2]
        # project the lcd screen to (wP,hP) size, make a imgWarp for the next step
        imgWarp = warpImg(contours_img, minAreaRectBox, wP, hP)
        if display: cv.imshow(windowName+' warped_ROI', imgWarp)

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
    imgTargetGray = img1  #cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    imgTemplateGray = img2  #cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(imgTemplateGray, None)
    kp2, des2 = orb.detectAndCompute(imgTargetGray, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv.drawMatches(imgTemplateGray, kp1, imgTargetGray, kp2,  matches[:maxMatchedCount], imgTargetGray, flags=2)
    if display:
        cv.imshow('matched window', img3)

def loadTrainImgDescriptors(folderName,featureFolderName,imgSuffix='png',descSuffix='npy'):
    """
    load train Info from specified folder into a dictionary
    :param folderName: raw image files dirname
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
    for dirpath, dirname, filenames in walk(folderName):
        for file in filenames:
            if file.split('.')[1] == imgSuffix:
                try:    # get raw image filename
                    fileId = int(file.split('.')[0])
                except ValueError:
                    # this is a ROI image file, create key,items in dictionary
                    key = file.split('.')[0].split('_')[0]
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
                    print('invalid npy filename:{}'.format(file))
                    raise ValueError
                else: # valid descriptor file, load it to respective key as value
                    key = file.split('.')[0]
                    trainInfoDict[key]['descriptor'] = np.load(featureFolderName + file)

    if len(trainInfoDict) == 0:
        return False, None
    else:
        return True, trainInfoDict


def queryDatabaseByFLANN(trainInfoDict, liveImgWarp, minMatchCount = 15):
    """
     compare liveImgWarp descriptor with all descriptors in trainImgInfos,
     return True and the most matched filename in trainImginfos, or False and None
    :param trainInfoDict:{
            'filename1':{'roiFileName':roifilename1,'descriptor': desc1},
            'filename2':{'roiFileName':roifilename2,'descriptor': desc2},
            ...
            }

    :param liveImgWarp:
    :param minMatchCount : minimumMatchedCount, if less than this, not match
    :return: True,filename or False,None
    """
    maxMatchCount = 0
    matchedName = None
    for key in trainInfoDict:
        matchedCount = matchByFLANN(liveImgWarp, trainInfoDict[key]['descriptor'])
        print('key={},matchedCount={},maxMatchCount={}'.format(key,matchedCount,maxMatchCount))
        if matchedCount >= minMatchCount and matchedCount >= maxMatchCount:
            maxMatchCount = matchedCount
            matchedName = key
    if matchedName is None:
        return False, None
    else:
        print('maxMatchCount={}, sample {} MATCHED!!'.format(maxMatchCount, matchedName))
        return True, matchedName


def matchByFLANN(targetImg, queryDescriptor):
    """
    compare both descriptors to get length of matched descriptors
    :param targetImg: warpedImg to be identified
    :param queryDescriptor:
    :return: length of matched descriptors
    """
    # generate both descriptors
    siftFeatureDetector = cv.xfeatures2d.SIFT_create()
    kp1, targetDescriptor = siftFeatureDetector.detectAndCompute(targetImg, None)


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

    return len(good)



cameraResW = 800
cameraResH = 600
scale = 2
wP = 300*scale
hP = 300*scale

import argparse,os.path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sample main utility to check lcd display")
    parser.add_argument("--device", dest='deviceId', help='video camera ID [0,1,2,3]',default=0, type=int)
    parser.add_argument("--liveMode", dest='webCamera', help='use camera to capture (1) or simulate (0)', default=1, type=int)
    args = parser.parse_args()
    print(args)

    saveImage = False
    drawRect = False
    liveCaptureWindowName = 'live_capture'
    expectedId = 18  # simulate the id of currently capture image
    success, queryImgDataBase = loadTrainImgDescriptors(r'./pictures/', r'./features/')
    if not success:
        print('training set loading failed, please check they are valid and retry!')
        exit(-1)

    camera = cv.VideoCapture(args.deviceId)
    camera.set(3, cameraResW)
    camera.set(4, cameraResH)
    # originalFileName = 'originLiveCapture.png'

    monitor = True
    if args.webCamera == 0:
        live_img = cv.imread(os.path.join('./pictures/', '9.png'))
        if live_img is not None:
            success = True
            liveCaptureWindowName += ' (playback mode)'

    while monitor:
        if args.webCamera == 1:
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
                                            threshold_2 = 112, kernelSize = 5, minArea = 50000, maxArea = 99502,
                                            windowName=liveCaptureWindowName)
        if liveFound:
            # result = matchByTemplate(imgWarp, liveImgWarp, matchThreshold=0.8, draw=True, save=saveImage)
            # matchByTemplate method does not fit for this scenario, since it requires pixel by pixel exact match ,
            # which is hard in engineering environment, the image captured  by camera are pixel-different even in same environment
            # in 2 shots
            result, matchedFileName = queryDatabaseByFLANN(queryImgDataBase, liveImgWarp, minMatchCount = 10)
            if result:
                matchedTrainingRoiImg = cv.imread('./pictures/' +
                                                  queryImgDataBase[matchedFileName]['roiFileName'])
                matchByORB(matchedTrainingRoiImg, liveImgWarp,
                           maxMatchedCount=500, display=True)       # draw matching points between 2,visualize them
                verdict = 'expected={}, matched={} '.format(expectedId, matchedFileName)
                print(verdict)
                cv.putText(matchedTrainingRoiImg,"sample id: " + matchedFileName,(10,500),
                           cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
                cv.imshow('matched training sample', matchedTrainingRoiImg)

                if expectedId == int(matchedFileName.split('.')[0]):
                    result = True
                else:
                    result = False
            else:   # not a match in training sets
                verdict = 'expected={}, no match '.format(expectedId)
                result = False
            # print('match result = {} '.format(result))
            markImg = live_img.copy()
            if result:
                cv.putText(markImg, verdict + ' Pass', (10,500),
                                    cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
            else:
                cv.putText(markImg, verdict + ' Fail', (10,500),
                           cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            cv.imshow(liveCaptureWindowName, markImg)

        else:
            markImg = live_img.copy()
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
        elif k in range(ord('1'),ord('9'),1): # simulate expected id
            expectedId = int(chr(k))

    cv.destroyAllWindows()

