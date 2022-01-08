"""
utility to try out different blur level and threshold level
in order to get a good contour rectangle

press 'a' key to save current images(original,detected,warped_image) to disk for late checking
press esc key to exit


Jan 5 , add Structure Similarity method tryout based on https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
 paper: https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
"""
import cv2
import numpy as np
from edgeDetect import getRequiredContours,warpImg
import subprocess
from managers import CommunicationManager, STATES_NUM
from train import SKIP_STATE_ID
import logging
from skimage.metrics import structural_similarity as compare_ssim
# import imutils
import time

def func(x):
    pass


STATES_NUM = 52
cameraResW = 1920
cameraResH = 1080
scale = 1
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

    screenResolution = ()
    screenResolution =_getCurrentScreenRes()
    _onDisplayId = 1


    # set up a mask to select interested zone only, discard 10 rows and 10 columns at 4 edges
    interestedMask = np.zeros((hP, wP), np.uint8)
    interestedMask[10:hP-10, 10:wP-10] = np.uint8(255)
    try:
        _communicationManager = CommunicationManager(logger, '/dev/ttyUSB'+str(0), 5, 1)
    except ConnectionError:
        logger.error('Abort!! Please make sure serial port is ready then retry')
        exit(-1)
    webCam = True
    # cameraProperty = []
    if webCam:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print('can not open camera 0')
            exit(-1)
        camera.set(3,cameraResW)
        camera.set(4,cameraResH)
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1)
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
        originalFileName = "./pictures/6.png"
        img = cv2.imread(originalFileName)

    windowName = 'looking for main Contour image'
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    saveLcdScreenMarked = False
    saveWarpImg = False
    saveLcdInternal = False
    drawRect = False

    roi_box = [(SX, SY), (LB_X, LB_Y), (EX, EY), (RU_X, RU_Y)]
    # normalize coordinates to integers
    box = np.int0(roi_box)

    tbCameraShiftX = 'cameraShiftX distance'   # simulate camera offset in x direction
    tbCameraShiftY = 'cameraShiftY distance'
    tbDiffThresh = 'Diff threshold'
    tbSSIMDiffThresh = 'SSIM diff judgeThreshold'       # threshold value to distinguish inter-frame difference
    tbThresh = 'threshold'
    tbErodeIter = 'erode iteration'
    tbDilateIter = 'dilate iteration'
    tbCannyLow = 'Canny low'
    tbCannyHigh = 'Canny high'

    cv2.createTrackbar(tbCameraShiftX, windowName, 0, 50, func)
    cv2.createTrackbar(tbCameraShiftY, windowName, 0, 50, func)
    cv2.createTrackbar(tbSSIMDiffThresh,windowName, 0, 255, func)
    cv2.createTrackbar(tbDiffThresh, windowName, 0, 255, func)
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

    switch = '0 : Origin\n1 : Gaussianblur\n2 : Thresh\n 3: Erode first\n 4: dilate first\n ' \
             '5:diff\n6:erode\n 7: SSIM warpImg diff \n 8:Contour\n'
    cv2.createTrackbar(switch, windowName, 0, 8, func)
    cv2.setTrackbarPos(switch, windowName, 7)
    cv2.setTrackbarPos(tbCameraShiftX, windowName, 0)
    cv2.setTrackbarPos(tbCameraShiftY, windowName, 0)
    cv2.setTrackbarPos(tbSSIMDiffThresh,windowName, 128)
    cv2.setTrackbarPos(tbDiffThresh, windowName, 18)
    cv2.setTrackbarPos(tbThresh, windowName, 65)
    cv2.setTrackbarPos(tbBlurlevel, windowName, 4)
    cv2.setTrackbarPos(tbMinArea, windowName, 50000)
    cv2.setTrackbarPos(tbMaxArea, windowName, 116230)
    cv2.setTrackbarPos(tbErodeIter, windowName, 1)
    cv2.setTrackbarPos(tbDilateIter, windowName, 1)
    cv2.setTrackbarPos(tbCannyLow, windowName, 10)
    cv2.setTrackbarPos(tbCannyHigh, windowName, 50)
    kernel = np.ones((3, 3))
    backGroundGray = None
    if backGroundGray is None:   # get the first frame , after gaussian blur, used as benchmark standard
        backGroundGray = cv2.cvtColor(cv2.imread('/media/newdiskp1/picMatch/trainingImages/52/pos-0.png', cv2.IMREAD_UNCHANGED),
                                      cv2.COLOR_BGR2GRAY)
    while(1):
        if webCam:
            success, img = camera.read()
            if not success:
                break

        thresh_level = cv2.getTrackbarPos(tbThresh, windowName)
        camera_shift_x = cv2.getTrackbarPos(tbCameraShiftX, windowName)
        camera_shift_y = cv2.getTrackbarPos(tbCameraShiftY, windowName)
        diff_thresh_level = cv2.getTrackbarPos(tbDiffThresh, windowName)
        diff_thresh_level = cv2.getTrackbarPos(tbDiffThresh, windowName)
        ssim_diff_thresh_level = cv2.getTrackbarPos(tbSSIMDiffThresh,windowName)
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
        backGroundGrayBlur = cv2.GaussianBlur(backGroundGray, (blurr_level, blurr_level), 0)

        # if backGroundGray is None:   # get the first frame , after gaussian blur, used as benchmark standard
        #     backGroundGray = gray
        #     backGroundGrayBlur = cv2.GaussianBlur(backGroundGray, (blurr_level, blurr_level), 0)
        #     continue



        if s == 0:
            displayWindow(windowName,img,30,0,screenResolution, True)
        elif s == 1:
            blur = cv2.GaussianBlur(gray, (blurr_level,blurr_level), 0)
            cv2.putText(blur, 'level={}'.format(blurr_level), (10,30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
            print('blurr_level={}'.format(blurr_level))
            displayWindow(windowName, blur, 30,0,screenResolution, True)
        elif s == 2:
            blur = cv2.GaussianBlur(gray,(blurr_level,blurr_level), 0)
            cv2.putText(blur, 'level={}'.format(blurr_level), (10,30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
            print('blurr_level={}'.format(blurr_level))
            ret, thresh_img = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY_INV)  #cv2.THRESH_TOZERO) #
            displayWindow(windowName,thresh_img, 30,0,screenResolution, True)
            # ret, thresh_img = cv2.threshold(gray, thresh_level,255, cv2.THRESH_BINARY_INV)
            # blur = cv2.blur(thresh_img,(blurr_level,blurr_level))

        # elif s == 3:  canny
        #     # blur = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 1)
        #     blur = cv2.blur(gray,(blurr_level,blurr_level))
        #     # retval, thresh = cv2.threshold(blur, erodeIter, 255, cv2.THRESH_BINARY)#|cv2.THRESH_OTSU)
        #     imgCanny = cv2.Canny(blur, erodeIter, dilateIter)
        #     displayWindow(windowName,imgCanny,30,0,screenResolution, True)
        #     # cv2.imshow (windowName, imgCanny)
        elif s == 3:
            blur = cv2.GaussianBlur(gray,(blurr_level,blurr_level), 0)
            cv2.putText(blur, 'level={}'.format(blurr_level), (10,30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
            print('blurr_level={}'.format(blurr_level))

            ret, thresh_img = cv2.threshold(blur, thresh_level,255, cv2.THRESH_BINARY_INV)
            imgErode = cv2.erode(thresh_img, kernel=kernel, iterations=erodeIter)
            imgDilate = cv2.dilate(imgErode, kernel=kernel, iterations=dilateIter)
            displayWindow(windowName, imgDilate,0,0,screenResolution, True)
        elif s == 4:
            blur = cv2.GaussianBlur(gray, (blurr_level,blurr_level),0)
            ret, thresh_img = cv2.threshold(blur, thresh_level,255, cv2.THRESH_BINARY_INV)

            imgDilate = cv2.dilate(thresh_img, kernel=kernel, iterations=dilateIter)
            imgErode = cv2.erode(imgDilate, kernel=kernel, iterations=erodeIter)
            displayWindow(windowName,imgErode,0,0,screenResolution, True)

            # # blur = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 1)
            # blur = cv2.blur(gray,(blurr_level,blurr_level))
            # imgCanny = cv2.Canny(blur, erodeIter, dilateIter)
            # imgDilate = cv2.dilate(imgCanny, kernel=kernel, iterations=3)
            # displayWindow(windowName,imgDilate,0,0,screenResolution, True)
            # cv2.imshow (windowName, imgDilate)
        elif s == 5:  # show difference of current frame against the previous frame under current setting
            blurFrame = cv2.GaussianBlur(gray, (blurr_level,blurr_level), 0)
            diff = cv2.absdiff(blurFrame, backGroundGrayBlur)
            if diff.max() <= diff_thresh_level:
                blurFrame = backGroundGrayBlur      #  treat current frame as the same as backGroundGrayBlur
            diff = cv2.absdiff(blurFrame, backGroundGrayBlur)
            diff = cv2.threshold(diff, diff_thresh_level, 255, cv2.THRESH_BINARY)[1]
            # imgDilateDiff = cv2.dilate(diff, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,4)), iterations=dilateIter)
            displayWindow(windowName, diff, 0,0,screenResolution, True)
            # set current frame as previous frame for next time
            # backGroundGrayBlur = cv2.GaussianBlur(gray, (blurr_level,blurr_level), 0)
        elif s == 6:
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
        elif s == 7: #  show SSIM difference of current frame against the previous frame under current setting
            # get the lcd Screen part rectangle from current frame after blur
            # and previous frame after blur
            # cv2.imshow('training',backGroundGrayBlur)
            blurFrame = cv2.GaussianBlur(gray, (blurr_level,blurr_level), 0)
            # setup  the simulated interested box after camera is shifted in x and y direction
            boxCameraShift = box + [camera_shift_x, camera_shift_y]
            imgWarpBlur = warpImg(blurFrame, boxCameraShift, wP, hP)
            # training sample's camera box does NOT change
            imgWarpBackgroundBlur = warpImg(backGroundGrayBlur, box,wP,hP)
            # cv2.imshow('warptraining',imgWarpBackgroundBlur)

            # use interestedMask to fetch only interested area data
            imgWarpBlur = np.where(interestedMask == 0, 0, imgWarpBlur)
            imgWarpBackgroundBlur = np.where(interestedMask == 0, 0, imgWarpBackgroundBlur)

            #calculate structure similarity
            score, diff = compare_ssim(imgWarpBlur, imgWarpBackgroundBlur, full=True)
            logger.debug('warp imgs structure similarity score ={}'.format(score))
            diff = (diff*255).astype('uint8')
            #displayWindow(windowName, diff, 0, 0, screenResolution, True)

            #find contours of difference between 2  warpImages
            thresh = cv2.threshold(diff, 255-ssim_diff_thresh_level, 255, cv2.THRESH_BINARY_INV)[1]  #cv2.THRESH_OTSU
            cnts, hiearachy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 0:
                logger.info('different images found! length of cnts={}'.format(len(cnts)))

            displayWindow(windowName, thresh, 0, 0, screenResolution, True)
            # cnts = imutils.grab_contours(cnts)

            # set current frame as previous frame for next time
            # backGroundGrayBlur = cv2.GaussianBlur(gray, (blurr_level, blurr_level), 0)
        else:
            # get the lcd Screen part rectangle from original img
            contours_img = img.copy()
            imgWarp = warpImg(contours_img, box, wP, hP)
            cv2.imshow('warped lcd screen img', imgWarp)

            # blurredImg, conts, contours_img = getRequiredContours(contours_img, blurr_level, cannyLow,cannyHigh,
            #                                                       erodeIter, dilateIter,
            #                                           kernel,
            #                                           minArea=minArea, maxArea=maxArea,
            #                                           cornerNumber=4, draw=drawRect,
            #                                           returnErodeImage=False)

            displayWindow(windowName,contours_img,0,0,screenResolution, True)
            # cv2.imshow(windowName, contours_img)
            # if len(conts) != 0:
            #     print('hhhh,conts = {}'.format(conts))
            #     minAreaRectBox = conts[0][2]
            #     # project the lcd screen to (wP,hP) size, make a imgWarp for the next step
            #     imgWarp = warpImg(blurredImg, minAreaRectBox, wP, hP)
            #     cv2.imshow('warped lcd screen img', imgWarp)
            #
            #     if saveLcdScreenMarked:
            #         cv2.imwrite(originalFileName.split('.')[0] +'_detected.png', contours_img)
            #         saveLcdScreenMarked = False
            #     if saveWarpImg:
            #         cv2.imwrite(originalFileName.split('.')[0] + '_warpImg.png', imgWarp)
            #         saveWarpImg = False

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

#        for i, key in enumerate(cameraPropertyNames):
#            # cameraProperty.append(camera.get(key))
#            print("{}={} ".format(cameraPropertyNames[key], camera.get(key)), end=' ')
#        print('\n')
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
