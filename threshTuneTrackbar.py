"""
utility to try out different blur level and threshold level
in order to get a good contour rectangle

press 'a' key to save current images(original,detected,warped_image) to disk for late checking
press esc key to exit
"""
import cv2
import numpy as np
from edgeDetect import getRequiredContours,warpImg
import subprocess
from managers import CommunicationManager, STATES_NUM
from train import SKIP_STATE_ID
import logging

def func(x):
    pass


STATES_NUM += 6
cameraResW = 1920
cameraResH = 1080
scale = 2
wP = 300*scale
hP = 300*scale

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

def displayWindow(windowName,frame, x, y, screenResolution,resize=False):
    """
    :param windowName:
    :param frame:
    :param x:
    :param y:
    :param screenResolution:
    :param resize: if resize to window to fit into display screen ratio, window takes 1/4 screen
    :return: width,height after resizing
    """
    cv2.moveWindow(windowName, x, y)
    width = int(frame.shape[1])
    height = int(frame.shape[0])
    if resize:
        scaleX = screenResolution[0]/frame.shape[1]/1.2
        scaleY = screenResolution[1]/frame.shape[0]/1.2
        scale = min(scaleX, scaleY)
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        cv2.resizeWindow(windowName, width, height)

    cv2.imshow(windowName, frame)
    return width, height

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s %(message)s')
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logHandler)

    screenResolution = ()
    screenResolution =_getCurrentScreenRes()
    _onDisplayId = 1
    try:
        _communicationManager = CommunicationManager(logger, '/dev/ttyUSB'+str(0), 5)
    except ConnectionError:
        logger.error('Abort!! Please make sure serial port is ready then retry')
        exit(-1)
    webCam = True
    if webCam:
        camera = cv2.VideoCapture(0)
        camera.set(3,cameraResW)
        camera.set(4,cameraResH)
        originalFileName = 'originLiveCapture.png'

    else:
        originalFileName = "./pictures/6.png"
        img = cv2.imread(originalFileName)

    windowName = 'looking for main Contour image'
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    saveLcdScreenMarked = False
    saveWarpImg = False
    saveLcdInternal = False
    drawRect = False

    tbThresh = 'threshold'
    tbErodeIter = 'erode iteration'
    tbDilateIter = 'dilate iteration'
    tbCannyLow = 'Canny low'
    tbCannyHigh = 'Canny high'
    cv2.createTrackbar(tbThresh, windowName, 0, 255, func)
    cv2.createTrackbar(tbErodeIter, windowName, 0, 5, func)
    cv2.createTrackbar(tbDilateIter, windowName, 0, 5, func)
    cv2.createTrackbar(tbCannyLow, windowName, 0, 255, func)
    cv2.createTrackbar(tbCannyHigh, windowName, 0, 255, func)
    tbBlurlevel = 'blur level'
    cv2.createTrackbar(tbBlurlevel, windowName, 1, 50, func)

    tbMinArea = 'minArea'
    tbMaxArea = 'maxArea'
    cv2.createTrackbar(tbMinArea, windowName, 20, 50000, func)
    cv2.createTrackbar(tbMaxArea, windowName, 20, 300000, func)

    switch = '0 : Origin\n1 : Thresh\n2 : blur\n 3: Erode first\n 4: dilate first\n 5:erode\n 6:Contour\n'
    cv2.createTrackbar(switch, windowName, 0, 6, func)
    cv2.setTrackbarPos(switch, windowName, 3)
    cv2.setTrackbarPos(tbThresh, windowName, 90)
    cv2.setTrackbarPos(tbBlurlevel, windowName, 4)
    cv2.setTrackbarPos(tbMinArea, windowName, 50000)
    cv2.setTrackbarPos(tbMaxArea, windowName, 116230)
    cv2.setTrackbarPos(tbErodeIter, windowName, 1)
    cv2.setTrackbarPos(tbDilateIter, windowName, 1)
    cv2.setTrackbarPos(tbCannyLow, windowName, 10)
    cv2.setTrackbarPos(tbCannyHigh, windowName, 50)
    kernel = np.ones((3, 3))
    while(1):
        if webCam:
            success, img = camera.read()
            if not success:
                break
        thresh_level = cv2.getTrackbarPos(tbThresh, windowName)
        cannyLow = cv2.getTrackbarPos(tbCannyLow, windowName)
        cannyHigh = cv2.getTrackbarPos(tbCannyHigh, windowName)
        erodeIter = cv2.getTrackbarPos(tbErodeIter, windowName)
        dilateIter = cv2.getTrackbarPos(tbDilateIter, windowName)
        blurr_level = cv2.getTrackbarPos(tbBlurlevel, windowName)
        blurr_level = (lambda x: x+1 if x % 2 == 0 else x)(blurr_level)    # avoid even

        minArea = cv2.getTrackbarPos(tbMinArea, windowName)
        maxArea = cv2.getTrackbarPos(tbMaxArea, windowName)
        s = cv2.getTrackbarPos(switch, windowName)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if s == 0:
            displayWindow(windowName,img,30,0,screenResolution, True)
            # cv2.imshow(windowName, img)
        elif s == 1:
            ret, thresh_img = cv2.threshold(gray, thresh_level,255,cv2.THRESH_TOZERO)#cv2.THRESH_BINARY_INV)
            displayWindow(windowName,thresh_img, 30,0,screenResolution, True)
        elif s == 2:
            ret, thresh_img = cv2.threshold(gray, thresh_level,255, cv2.THRESH_BINARY_INV)
            blur = cv2.blur(thresh_img,(blurr_level,blurr_level))
            cv2.putText(blur, 'level={}'.format(blurr_level), (10,30),
                       cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
            print('blurr_level={}'.format(blurr_level))
            displayWindow(windowName,thresh_img,30,0,screenResolution, True)
            # cv2.imshow(windowName, blur)
        # elif s == 3:  canny
        #     # blur = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 1)
        #     blur = cv2.blur(gray,(blurr_level,blurr_level))
        #     # retval, thresh = cv2.threshold(blur, erodeIter, 255, cv2.THRESH_BINARY)#|cv2.THRESH_OTSU)
        #     imgCanny = cv2.Canny(blur, erodeIter, dilateIter)
        #     displayWindow(windowName,imgCanny,30,0,screenResolution, True)
        #     # cv2.imshow (windowName, imgCanny)
        elif s == 3:
            ret, thresh_img = cv2.threshold(gray, thresh_level,255, cv2.THRESH_BINARY_INV)
            blur = cv2.blur(thresh_img,(blurr_level,blurr_level))
            imgErode = cv2.erode(blur, kernel=kernel, iterations=erodeIter)
            imgDilate = cv2.dilate(imgErode, kernel=kernel, iterations=dilateIter)
            # imgErode = cv2.erode(imgDilate, kernel=kernel, iterations=2)
            displayWindow(windowName,imgDilate,0,0,screenResolution, True)
        elif s == 4:
            ret, thresh_img = cv2.threshold(gray, thresh_level,255, cv2.THRESH_BINARY_INV)
            blur = cv2.blur(thresh_img,(blurr_level,blurr_level))
            imgDilate = cv2.dilate(blur, kernel=kernel, iterations=dilateIter)
            imgErode = cv2.erode(imgDilate, kernel=kernel, iterations=erodeIter)
            # imgErode = cv2.erode(imgDilate, kernel=kernel, iterations=2)
            displayWindow(windowName,imgDilate,0,0,screenResolution, True)

            # # blur = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 1)
            # blur = cv2.blur(gray,(blurr_level,blurr_level))
            # imgCanny = cv2.Canny(blur, erodeIter, dilateIter)
            # imgDilate = cv2.dilate(imgCanny, kernel=kernel, iterations=3)
            # displayWindow(windowName,imgDilate,0,0,screenResolution, True)
            # cv2.imshow (windowName, imgDilate)
        elif s == 5:
            # show erodeImage
            contours_img = img.copy()
            erodeImage, conts, contours_img = getRequiredContours(contours_img, blurr_level, cannyLow,cannyHigh,
                                                    erodeIter, dilateIter,
                                                    kernel,
                                                    minArea=minArea, maxArea=maxArea,
                                                    cornerNumber=4, draw=drawRect,
                                                    returnErodeImage=True,threshLevel=thresh_level)
            displayWindow(windowName,erodeImage,0,0,screenResolution, True)
            # cv2.imshow(windowName, erodeImage)
            if len(conts) != 0:
                # print('hhhh,conts = {}'.format(conts))
                minAreaRectBox = conts[0][2]
                # project the lcd screen to (wP,hP) size, make a imgWarp for the next step
                imgWarp = warpImg(erodeImage, minAreaRectBox, wP, hP)
                cv2.imshow('warped erode img', imgWarp)
            # end of getcontours
        else:
            # get the lcd Screen part rectangle from original blurred img
            contours_img = img.copy()
            blurredImg, conts, contours_img = getRequiredContours(contours_img, blurr_level, cannyLow,cannyHigh,
                                                                  erodeIter, dilateIter,
                                                      kernel,
                                                      minArea=minArea, maxArea=maxArea,
                                                      cornerNumber=4, draw=drawRect,
                                                      returnErodeImage=False)
            displayWindow(windowName,contours_img,0,0,screenResolution, True)
            # cv2.imshow(windowName, contours_img)
            if len(conts) != 0:
                print('hhhh,conts = {}'.format(conts))
                minAreaRectBox = conts[0][2]
                # project the lcd screen to (wP,hP) size, make a imgWarp for the next step
                imgWarp = warpImg(blurredImg, minAreaRectBox, wP, hP)
                cv2.imshow('warped lcd screen img', imgWarp)

                if saveLcdScreenMarked:
                    cv2.imwrite(originalFileName.split('.')[0] +'_detected.png', contours_img)
                    saveLcdScreenMarked = False
                if saveWarpImg:
                    cv2.imwrite(originalFileName.split('.')[0] + '_warpImg.png', imgWarp)
                    saveWarpImg = False

        # getRequiredContours from imgWarp, looking for symbols displayed on lcd screen
        #         contours_img2, conts2 = \
        #             getRequiredContours(imgWarp, blurr_level, erodeIter, dilateIter,
        #                               kernel,
        #                               minArea=4, maxArea=minArea,
        #                               cornerNumber=4, draw=True,
        #                               needPreProcess=False
        #                               )
        #
        #         if len(conts2) != 0:
        #             for symbol in conts2:
        #                 cv2.polylines(contours_img2, [symbol[1]], True, (255, 0, 0), 2)  #draw symbol's approx in BLUE
        #             cv2.imshow('lcd internal window', contours_img2)
        #             if saveLcdInternal:
        #                 cv2.imwrite(originalFileName.split('.')[0] + '_insideDetectImg.png', contours_img2)
        #                 saveLcdInternal = False
        #


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
        elif k == ord('n') or k == ord('N'):  # simulate move DUT to next image
            _onDisplayId += 1
            if _onDisplayId == STATES_NUM+1:
                _onDisplayId = 1
            elif _onDisplayId == SKIP_STATE_ID:
                _onDisplayId += 1
            command = _communicationManager.send(_onDisplayId)
            response = _communicationManager.getResponse()
            if response[:-1] == command[:]:
                print('===>>> get valid response from DUT,\nDUT moves to next image type {}'.format(_onDisplayId))
        elif k == ord('b') or k == ord('B'):  # simulate move DUT to previous image
            _onDisplayId -= 1
            if _onDisplayId == 0:
                _onDisplayId = STATES_NUM
            elif _onDisplayId == SKIP_STATE_ID:
                _onDisplayId -= 1
            command = _communicationManager.send(_onDisplayId)
            response = _communicationManager.getResponse()
            if response[:-1] == command[:]:
                print('===>>> get valid response from DUT,\nDUT moves to previous image type {}'.format(_onDisplayId))





    cv2.destroyAllWindows()
    camera.release()
