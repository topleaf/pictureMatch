"""
utility to try out different blur level and threshold level
in order to get a good contour rectangle , horizontal lines , OCR on test tube

based on previous threshTuneTrackbar.py

press 'a' key to save current images(original,detected,warped_image) to disk for late checking
press esc key to exit


start date: June 15,2022

"""
import cv2
import numpy as np
from edgeDetect import getRequiredContours, warpImg, getRequiredContoursByThreshold
import subprocess
import logging
# import imutils
import time

def func(x):
    pass


cameraResW = 1920
cameraResH = 1080
scale = 2
wP = 300*scale
hP = 300*scale

from train import SX, SY, EX, EY, RU_X, RU_Y, LB_X, LB_Y

def _getCurrentScreenRes():
    """
    in linux, use 'xrandr|grep \*' to get current system's screen resolution
    pipeline is used
    $ xrandr|grep \*
       1600x900      59.99*   59.94    59.95    59.82

    :return: width,height in int
    """
    cmd = ['xrandr']
    cmd2 =['grep', '\*']
    p = subprocess.Popen(cmd,stdout=subprocess.PIPE)
    p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)

    out, junk = p2.communicate()
    p2.stdout.close()
    resolution = out.decode('utf-8')
    return int(resolution.split('x')[0]), int(resolution.split('x')[1].split(' ')[0])


def resizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def displayWindow(windowName,frame, x, y, screenResolution,resize=False):
    """
    :param windowName:
    :param frame:
    :param x: The new x-coordinate of the window.
    :param y: The new y-coordinate of the window.
    :param screenResolution:
    :param resize: if resize to window to fit into display screen ratio, window takes 1/4 screen
    :return: width,height after resizing
    """
    # cv2.moveWindow(windowName, x, y)
    width = int(frame.shape[1])
    height = int(frame.shape[0])
    if resize:
        scaleX = screenResolution[0]/frame.shape[1]/1.0
        scaleY = screenResolution[1]/frame.shape[0]/1.0
        scale = min(scaleX, scaleY)
        if scale == scaleY:    # y direction ratio is smaller than x direction ratio
            frame = resizeWithAspectRatio(frame, width=None, height=screenResolution[1], inter=cv2.INTER_AREA)
        else:
            frame = resizeWithAspectRatio(frame, width=screenResolution[0], height=None,inter=cv2.INTER_AREA)

    # automatically resizeWindow to hold current frame inside window completely, this is optional
    cv2.resizeWindow(windowName, frame.shape[1], frame.shape[0])
    cv2.imshow(windowName, frame)

    return width, height

#
# def displayWindow(windowName,frame, x, y, screenResolution,resize=False):
#     """
#     :param windowName:
#     :param frame:
#     :param x: The new x-coordinate of the window.
#     :param y: The new y-coordinate of the window.
#     :param screenResolution:
#     :param resize: if resize to window to fit into display screen ratio, window takes 1/4 screen
#     :return: width,height after resizing
#     """
#     #cv2.moveWindow(windowName, x, y)
#     width = int(frame.shape[1])
#     height = int(frame.shape[0])
#     if resize:
#         scaleX = screenResolution[0]/frame.shape[1]/1.0
#         scaleY = screenResolution[1]/frame.shape[0]/1.0
#         scaleMin = min(scaleX, scaleY)
#         width = int(frame.shape[1]*scaleMin)
#         height = int(frame.shape[0]*scaleMin)
#     #    cv2.resizeWindow(windowName, width, height)
#         frame = cv2.resize(frame,(width,height))
#         logger.info("new width={},height={}".format(width,height))
#
#     cv2.imshow(windowName, frame)
#     return width, height

cameraPropertyNames={
    cv2.CAP_PROP_FRAME_WIDTH:'width',
    cv2.CAP_PROP_FRAME_HEIGHT:'height',
    cv2.CAP_PROP_BUFFERSIZE:'BufferSize',
    cv2.CAP_PROP_BRIGHTNESS:"Brightness",
    cv2.CAP_PROP_CONTRAST:"Contrast",
    cv2.CAP_PROP_SATURATION:"Saturation",
    cv2.CAP_PROP_HUE: "Hue",
    cv2.CAP_PROP_GAIN:'Gain',
    cv2.CAP_PROP_EXPOSURE:"Exposure",
    cv2.CAP_PROP_FOCUS: "Focus",
    cv2.CAP_PROP_APERTURE:'APerture',
    cv2.CAP_PROP_AUTO_EXPOSURE:'auto exposure',
    cv2.CAP_PROP_AUTO_WB:'auto WB',
    cv2.CAP_PROP_AUTOFOCUS: 'autoFocus',
    cv2.CAP_PROP_CONVERT_RGB:'convertRGB',
}

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s %(message)s')
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logHandler)

    screenResolution = (1280, 1024)  # my win7 PC's screenresolution
    # screenResolution =_getCurrentScreenRes()
    _onDisplayId = 1


    # set up a mask to select interested zone only,
    interestedMask = np.zeros((hP, wP), np.uint8)
    # discard 10 rows and 10 columns at 4 edges
    # interestedMask[10:hP-10, 10:wP-10] = np.uint8(255)
    interestedMask[0:hP, 0:wP] = np.uint8(255)
    webCam = False
    # cameraProperty = []
    if webCam:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print('can not open camera 0')
            exit(-1)
        camera.set(3,cameraResW)
        camera.set(4,cameraResH)
        # camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)
        # camera.set(cv2.CAP_PROP_EXPOSURE, -2)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # camera.set(cv2.CAP_PROP_HUE, 1.0)
        # camera.set(cv2.CAP_PROP_AUTO_WB, -1.0)
        # camera.set(cv2.CAP_PROP_SATURATION,50)
        camera.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        originalFileName = 'originLiveCapture.png'
        for i, key in enumerate(cameraPropertyNames):
            # cameraProperty.append(camera.get(key))
            print("{}={} ".format(cameraPropertyNames[key], camera.get(key)), end=' ')
        print('\n')
    else:
        originalFileName = "sample2_croped.jpg"
        img = cv2.imread(originalFileName)
        if img is None:
            logger.error("test image file name {} does not exist,please check and retry".format(originalFileName))
            exit(-1)

    windowName = 'looking for main Contour image'
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    trackbarWindowName = 'trackbars'
    cv2.namedWindow(trackbarWindowName, cv2.WINDOW_NORMAL)

    drawRect = True

    # roi_box = [(SX, SY), (LB_X, LB_Y), (EX, EY), (RU_X, RU_Y)]
    # normalize coordinates to integers
    # box = np.int0(roi_box)

    tbThresh = 'threshold'
    tbErodeIter = 'erode iteration'
    tbDilateIter = 'dilate iteration'
    tbCannyLow = 'Canny low'
    tbCannyHigh = 'Canny high'
    tbMinLineLength = 'minLineLength'
    tbMaxLineGap = 'maxLineGap'


    cv2.createTrackbar(tbThresh, trackbarWindowName, 55, 255, func)
#    cv2.createTrackbar(tbErodeIter, trackbarWindowName, 0, 5, func)

 #   cv2.createTrackbar(tbDilateIter, trackbarWindowName, 0, 5, func)
    cv2.createTrackbar(tbCannyLow, trackbarWindowName, 0, 255, func)
    cv2.createTrackbar(tbCannyHigh, trackbarWindowName, 0, 255, func)
    tbBlurlevel = 'blur level'
    cv2.createTrackbar(tbBlurlevel, trackbarWindowName, 1, 50, func)
    # kernel size for cv.morphologyEx close operation, the larger the kernel size,
    # adjacent shapes will be more likely to form into one shape to reduce noisy holes
    tbKernelSize = 'kernel size'
    cv2.createTrackbar(tbKernelSize, trackbarWindowName, 0, 50, func)

    tbMinArea = 'minArea'
    tbMaxArea = 'maxArea'
    cv2.createTrackbar(tbMinArea, trackbarWindowName, 475000, 500000, func)
    cv2.createTrackbar(tbMaxArea, trackbarWindowName, 3495000, 5500000, func)

    tbMinMarkArea = 'minAreaMark'       # mark line minimum area
    tbMaxMarkArea = 'maxAreaMark'       # mark line maximum area
    cv2.createTrackbar(tbMinMarkArea, trackbarWindowName, 20, 10000, func)
    cv2.createTrackbar(tbMaxMarkArea, trackbarWindowName, 300, 50000, func)
    cv2.createTrackbar(tbMinLineLength, trackbarWindowName , 5, 255, func)
    cv2.createTrackbar(tbMaxLineGap, trackbarWindowName , 20, 255, func)


#    switch = '0 : Origin\n1 : Gaussianblur\n2 : Thresh\n 3: waterlevel detection\n 4: canny\n ' \
 #            '5:line detection after canny\n 6: contour detection'
    switch = '0 :o 1 :blur 2 : Thresh 3: waterlevel 4: canny ' \
             '5:line dt 6:contour dt'
    cv2.createTrackbar(switch, trackbarWindowName, 0, 6, func)
    cv2.setTrackbarPos(switch, trackbarWindowName, 6)
    cv2.setTrackbarPos(tbThresh, trackbarWindowName, 54)
    cv2.setTrackbarPos(tbBlurlevel, trackbarWindowName, 4)
    cv2.setTrackbarPos(tbKernelSize, trackbarWindowName, 29)
    cv2.setTrackbarPos(tbMinArea, trackbarWindowName, 182077)
    cv2.setTrackbarPos(tbMaxArea, trackbarWindowName, 4032800)
    #cv2.setTrackbarPos(tbErodeIter, trackbarWindowName, 1)
    #cv2.setTrackbarPos(tbDilateIter, trackbarWindowName, 1)
    cv2.setTrackbarPos(tbCannyLow, trackbarWindowName, 220)
    cv2.setTrackbarPos(tbCannyHigh, trackbarWindowName, 254)

    while(1):
        if webCam:
            success, img = camera.read()
            if not success:
                break

        kernelSize = cv2.getTrackbarPos(tbKernelSize, trackbarWindowName)
        kernel = np.ones((kernelSize, kernelSize))
        thresh_level = cv2.getTrackbarPos(tbThresh, trackbarWindowName)
        cannyLow = cv2.getTrackbarPos(tbCannyLow, trackbarWindowName)
        cannyHigh = cv2.getTrackbarPos(tbCannyHigh, trackbarWindowName)
     #   erodeIter = cv2.getTrackbarPos(tbErodeIter, trackbarWindowName)
     #   dilateIter = cv2.getTrackbarPos(tbDilateIter, trackbarWindowName)
        blurr_level = cv2.getTrackbarPos(tbBlurlevel, trackbarWindowName)
        blurr_level = (lambda x: x+1 if x % 2 == 0 else x)(blurr_level)    # avoid even value


        minLineLength = cv2.getTrackbarPos(tbMinLineLength,trackbarWindowName)
        maxLineGap = cv2.getTrackbarPos(tbMaxLineGap,trackbarWindowName)
        minArea = cv2.getTrackbarPos(tbMinArea, trackbarWindowName)
        maxArea = cv2.getTrackbarPos(tbMaxArea, trackbarWindowName)
        minMarkArea = cv2.getTrackbarPos(tbMinMarkArea, trackbarWindowName)
        maxMarkArea = cv2.getTrackbarPos(tbMaxMarkArea, trackbarWindowName)
        s = cv2.getTrackbarPos(switch, trackbarWindowName)


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        if s == 0:  # show original image
            displayWindow(windowName, img, 30, 0, screenResolution, True)
        elif s == 1:  # show blurred image
            blur = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 0)
            cv2.putText(blur, 'level={}'.format(blurr_level), (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
            # print('blurr_level={}'.format(blurr_level))
            displayWindow(windowName, blur, 30, 0, screenResolution, True)
        elif s == 2:        # show threshold image
            blur = cv2.GaussianBlur(gray, (blurr_level,blurr_level), 0)
            cv2.putText(blur, 'level={}'.format(blurr_level), (10,30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
            ret, thresh_img = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY_INV)  #cv2.THRESH_TOZERO) #
            # apply close operation to thresh_img
            # 先膨胀再腐蚀。被用来填充前景物体中的小洞，或者前景物体上面的小黑点。使得前景变成更加连续的实体
            # IMPORTANT:
            # this operation can improve consistency of recognition algorithm,
            # because 1. it helps to connect closely-adjacent shapes to one shape, so it reduces contours number
            # 2, in later steps of  _compare, area calculation workload is reduced
            # the larger the kernel size, adjacent shapes will be more likely to form into one shape
            closed = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
            displayWindow(windowName, closed, 30, 0, screenResolution, True)

        elif s == 3:   # show original image with water level detection rectangle shown if there is any
            contours_img = img.copy()
            threshedImg, conts, contours_img = getRequiredContoursByThreshold(contours_img, blurr_level, thresh_level,
                                                    cv2.THRESH_BINARY_INV,
                                                    kernel,
                                                    minArea=minArea, maxArea=maxArea,
                                                    cornerNumber=4, draw=drawRect)
            displayWindow(windowName, contours_img, 0, 0, screenResolution, True)
        elif s == 4:  # show canny
            #  step 1: find out the rectangle of black contents in the test tube
            contours_img = img.copy()
            threshedImg, conts, contours_img = getRequiredContoursByThreshold(contours_img, blurr_level, thresh_level,
                                                                              cv2.THRESH_BINARY_INV,
                                                                              kernel,
                                                                              minArea=minArea, maxArea=maxArea,
                                                                              cornerNumber=4, draw=drawRect)
            if len(conts) == 0:
                logger.info("no bounding box found")
                displayWindow(windowName, contours_img, 0, 0, screenResolution, True)
            else:
                # the first boundingbox marks the black contents inside the test tube
                box = conts[0][0]
                blurFrame = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 0)
                #  step 2: projecting the whole black contents to a new coordination system [wP,hP]
                # where the water level's new y coordination is 0
                imgContentWarpBlur = warpImg(blurFrame, box, wP, hP)
                # imgContentWarp = warpImg(img, box, wP, hP)
                # step 3: canny it before applying line detection
                edges = cv2.Canny(imgContentWarpBlur, cannyLow, cannyHigh)
                displayWindow(windowName, edges, 0, 0, screenResolution, True)




            # dilate first, erode later, show result
            # blur = cv2.GaussianBlur(gray, (blurr_level,blurr_level),0)
            # ret, thresh_img = cv2.threshold(blur, thresh_level,255, cv2.THRESH_BINARY_INV)
            #
            # imgDilate = cv2.dilate(thresh_img, kernel=kernel, iterations=dilateIter)
            # imgErode = cv2.erode(imgDilate, kernel=kernel, iterations=erodeIter)
            # displayWindow(windowName,imgErode,0,0,screenResolution, True)


        elif s == 5:
            # to detect the first and second horizontal lines from top to bottom, get their y coordinations
            # using line detection API
            #  step 1: find out the rectangle of black contents in the test tube
            contours_img = img.copy()
            threshedImg, conts, contours_img = getRequiredContoursByThreshold(contours_img, blurr_level, thresh_level,
                                                                              cv2.THRESH_BINARY_INV,
                                                                              kernel,
                                                                              minArea=minArea, maxArea=maxArea,
                                                                              cornerNumber=4, draw=drawRect)
            if len(conts) == 0:
                logger.info("no bounding box found, show original image instead")
                displayWindow(windowName, contours_img, 0, 0, screenResolution, True)
            else:
                # the first boundingbox marks the black contents inside the test tube
                box = conts[0][0]
                blurFrame = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 0)
                #  step 2: projecting the whole black contents to a new coordination system [wP,hP]
                # where the water level's new y coordination is 0
                imgContentWarpBlur = warpImg(blurFrame, box, wP, hP)
                imgContentWarp = warpImg(img, box, wP, hP)
                # step 3: canny it before applying line detection
                edges = cv2.Canny(imgContentWarpBlur, cannyLow, cannyHigh)

                # step 4: draw detected lines on the original image in GREEN, Only straight lines are detected
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
                if lines is not None:
                    for x1, y1, x2, y2 in lines[0]:
                        cv2.line(imgContentWarp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    logger.info('no line detected.please tune cannyLow/High,minLineLength/maxLineGap ')

                displayWindow(windowName, imgContentWarp, 0, 0, screenResolution, True)
                # displayWindow(windowName, edges, 0, 0, screenResolution, True)
        elif s == 6:
            # to detect the first and second horizontal lines from top to bottom, get their y coordinations
            # using findContour API

            #  step 1: find out the rectangle of black contents in the test tube
            contours_img = img.copy()
            threshedImg, conts, contours_img = getRequiredContoursByThreshold(contours_img, blurr_level, thresh_level,
                                                                              cv2.THRESH_BINARY_INV,
                                                                              kernel,
                                                                              minArea=minArea, maxArea=maxArea,
                                                                              cornerNumber=4, draw=drawRect)
            if len(conts) == 0:
                logger.info("no bounding box found, show original image instead")
                displayWindow(windowName, contours_img, 0, 0, screenResolution, True)
            else:
                # the first boundingbox marks the black contents inside the test tube
                box = conts[0][0]
                blurFrame = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 0)
                #  step 2: projecting the whole black contents to a new coordination system [wP,hP]
                # where the water level's new y coordination is 0
                imgContentWarpBlur = warpImg(blurFrame, box, wP, hP)
                imgContentWarp = warpImg(img, box, wP, hP)
                # step 3: find the first block horizontally which might be the closest mark on test tube under water level.
                # here, threshold_type argument MUST be set to cv2.THRESH_BINARY, so that the white marks on test tube
                # are shown in WHITE for cv.findContours API to identify them as contours.
                threshedImg, conts, contours_img = getRequiredContoursByThreshold(imgContentWarp.copy(), blurr_level, thresh_level,
                                                                                  cv2.THRESH_BINARY,
                                                                                  kernel,
                                                                                  minArea=minMarkArea, maxArea=maxMarkArea,
                                                                                  cornerNumber=4, draw=drawRect)


                displayWindow(windowName, contours_img, 0, 0, screenResolution, True)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('d') or k == ord('D'):  # enable draw rectangle around contours
            drawRect = True
        elif k == ord('u') or k == ord('U'):  # disable draw rectangle around contours
            drawRect = False




    cv2.destroyAllWindows()
    if webCam:
        camera.release()
