
"""
Author: Jin Li
Date: Dec 2,2021
build query images database information,
1. capture all possible standard images from camera, save to disk
2. detect ROI (by corner , or by edgeDetection), warp image to 600*600 coordination system, save to disk
3. detect and compute their keypoints and featureDescriptor, save to disk

Date: Dec 12, 2021
change strategy: train STATE_NUM  svm models to predict if any given image is or is not detected as NO. STATE_NUM image
image id =1 ,whose positive trainingImages are stored in /trainImages/1/pos-1.png, pos-2.png .... , pos-100.png
image id =1 whose negative  trainingImages are stored in /trainImages/1/neg-1.png, neg-2.png .... , pos-x.png

...

image id =15 ,whose positive trainingImages are stored in /trainImages/15/pos-1.png, pos-2.png .... , pos-100.png
image id =15 whose negative  trainingImages are stored in /trainImages/15/neg-1.png, neg-2.png .... , pos-x.png

svm1, svm2, ... svm15 are 15 different models, trained to detect if any given image belongs to image 1, ... image15 or not

Date: Dec 23, 2021

use threshTuneTrackbar.py to search for best threshold value manually, use this threshed value to
create thresh image with blur (9,9) , use this image as target to detect keypoints and compute SIFT features ,

Date: Dec 24, 2021
train one multiple-classification SVM model instead of multiple 2 classification SVM models

Date: Dec 25,2021
preprocess improvement, after blur/threshold, use erode 1 and dilate 1 with kernel=np.ones((3,3)) to remove noise

Date: Dec 27,2021
warp interested Region image, detect and compute on warpedImage, use key 'r' to show warped images/keypoints
set videoCapture property buffersize from default 4 to 1
"""

from managers import WindowManager,CaptureManager,CommunicationManager, STATES_NUM,SKIP_STATE_ID
from edgeDetect import extractValidROI, warpImg
import logging
import argparse
from os import walk, mkdir,rmdir,remove,removedirs
from os.path import join
import cv2 as cv
import numpy as np
import time

DELAY_IN_SECONDS = 0.01



# define the absolute coordinations of the lcd screen in the image captured by camera, in pixel
SX, SY = 713, 237       # left top corner
EX, EY = 1292, 872      # right bottom corner
RU_X, RU_Y = 1292, 237  # right top corner
LB_X, LB_Y = 718, 872   # left bottom corner
#

class BuildDatabase(object):
    def __init__(self,logger,windowName,captureDeviceId,predefinedPatterns,portId,
                 duration, videoWidth, videoHeight, wP, hP, folderName,
                 imgFormat,skipCapture=True,
                 blurLevel=9,noiseLevel=8,imageTheme=3,structureSimilarityThreshold=23,
                 offsetX=5,offsetY=5,deltaArea=40,deltaCenterX=20,deltaCenterY=20,deltaRadius=10):
        """

        :param windowName: the title of window to show captured picture,str
        :param captureDeviceId:  the device id of the camera
        :param predefinedPatterns:
        :param portId: serial port id: int
        """
        #define roi mask coordinations in WarpImage
        self._S_MASKX, self._E_MASKX, self._S_MASKY, self._E_MASKY = 10, wP-10, 1, hP-1
        self.logger = logger
        self._compareResultList = []
        self._roi_box = [(SX, SY), (LB_X, LB_Y), (EX, EY), (RU_X, RU_Y)]

        # normalize coordinates to integers
        self.box = np.int0(self._roi_box)
        self._snapshotWindowManager = WindowManager("Training Sample", None)
        self._windowManager = WindowManager(windowName, self.onKeyPress)
        self._promptWindowManager = WindowManager('Press return to continue', None)  # act as a GUI user interface window

        self._captureManager = CaptureManager(logger, captureDeviceId,
                                              self._windowManager, self._snapshotWindowManager,
                                              self._promptWindowManager,
                                              False, videoWidth, videoHeight,
                                              self._compareResultList,
                                              warpImgSize=(wP, hP),
                                              blurLevel=blurLevel,
                                              roiBox=self.box,
                                              cameraNoise=noiseLevel,
                                              structureSimilarityThreshold=structureSimilarityThreshold,
                                              offsetRangeX=offsetX,
                                              offsetRangeY=offsetY,
                                              deltaArea=deltaArea, deltaCenterX=deltaCenterX,
                                              deltaCenterY=deltaCenterY, deltaRadius=deltaRadius
                                              )
        self._predefinedPatterns = predefinedPatterns
        self._expireSeconds = 5
        self._trainingFrameCount = duration
        self._warpImgWP, self._warpImgHP = wP, hP
        self._warpImgBox = [(self._S_MASKX, self._S_MASKY), (self._S_MASKX, self._E_MASKY),
                            (self._E_MASKX, self._E_MASKY), (self._E_MASKX, self._S_MASKY)]
        try:
            self._communicationManager = CommunicationManager(self.logger, '/dev/ttyUSB'+str(portId),
                                                              self._expireSeconds, algorithm=imageTheme)
        except ConnectionError:
            self.logger.error('Abort!! Please make sure serial port {} is available before retry'.format(str(portId)))
            exit(-1)
        self._windowManager.createWindow()
        self._snapshotWindowManager.createWindow()
        self.__testResults = []
        self._current = None
        self._waitFrameCount = 0  #
        self._folderName = folderName
        self._imgFormat = imgFormat

        self._skipCapture = skipCapture
        self.positive = 'pos-'      # positive training samples filename prefix
        # set up a mask to select interested zone only for further compare
        self.interestedMask = np.zeros((self._warpImgHP, self._warpImgWP), np.uint8)
        self.interestedMask[self._S_MASKY:self._E_MASKY, self._S_MASKX:self._E_MASKX] = np.uint8(255)
        self.expectedTrainingImageId = STATES_NUM  #  expected training sample id to be compared with current image
        self._onDisplayId = STATES_NUM
        self._expectedTrainingImg = None  # load expected svmmodel's first training image
        self._startTime = time.time()
        self._readyToNext = False       # OK to move to next round , test next LCD module

    def run(self):
        #  step 1: capture and save all positive training images to respective image folders/pos-x
        self._captureAndSaveAllImages()
        training_duration = time.time() - self._startTime
        self.logger.info('Training duration is:{} seconds'.format(training_duration))
        # step 2: loop to test if DUT's all images match with corresponding training images
        self._makeJudgement()

    def _path(self, typeid, cls, i):
        """
        get positive or negative image files full path
        :param typeid:  positive image type id
        :param cls: 'pos-' or 'neg-'
        :param i:  sample id in int  , 0 < i < STATES_NUM
        :return:
        """
        if cls == self.positive:
            return "%s/%s%d.%s" % (join(self._folderName, str(typeid)), cls, i, self._imgFormat)
        else:  # no neg- folder
            raise ValueError

    def _captureAndSaveAllImages(self):
        """
        capture training images and save them to self._folderName,
        :return:
        """
        try:
            mkdir(self._folderName)
        except FileExistsError as e:
            self.logger.debug('image folder %s exists' % self._folderName)
            # rmdir(self._folderName)
        if self._skipCapture:
            self.logger.debug('skip overwriting')
        else:
            for self._current in self._predefinedPatterns:
                if self._current == SKIP_STATE_ID:
                    self.logger.debug('skip capturing training sample of id = {}'.format(SKIP_STATE_ID))
                    continue
                response = b''
                retry = 0
                while response == b'' and retry < 3:
                    self.logger.debug('send command to switch to {},retry={}'.format(self._current, retry))
                    assert self._current in range(1, STATES_NUM+1)
                    command = self._communicationManager.send(self._current)
                    response = self._communicationManager.getResponse()
                    if response[:-1] == command[:]:
                        self.logger.info('===>>> get valid response, wait for {} second,\n'
                                         ' then start capturing and saving'
                                         ' positive training images for type {}'.format(DELAY_IN_SECONDS,self._current))
                        time.sleep(DELAY_IN_SECONDS)
                        self._captureManager.openCamera()

                        try:
                            newFolder = join(self._folderName, str(self._current))
                            mkdir(newFolder)
                        except FileExistsError as e:
                            self.logger.debug('OK. training image folder %s exists' %newFolder)

                        self._waitFrameCount = 0
                        while self._windowManager.isWindowCreated and self._waitFrameCount <= \
                                self._trainingFrameCount:
                            self._captureManager.enterFrame()
                            if self._waitFrameCount < self._trainingFrameCount:
                                # capture and save the next frame img to file
                                # in current directory by the name of pos-1.png, ..., pos-100.png
                                    self._captureManager.save(join(newFolder, self.positive + str(self._waitFrameCount) +
                                                              '.' + self._imgFormat))
                            self._waitFrameCount += 1
                            self._captureManager.exitFrame()
                            self._windowManager.processEvents()
                        if self._captureManager.cameraIsOpened:
                            self._captureManager.closeCamera()

                    elif response == b'':
                        self.logger.debug('no response after few seconds timeout')
                        retry += 1
                    else:
                        self.logger.error("Receive a mismatched response from controller,skip this pic :{}".format(response))
                        self.__testResults.append('O')
                        break

                if retry == 3:
                    self.logger.debug('pic_id:{},communication timeout,no response from controller after 3 times retry!'.format(self._current))
                    self.__testResults.append('N')
                    continue
        # load expected training sample
        firstExpectedTrainingFileLocation = self._path(self.expectedTrainingImageId, self.positive, 0)
        self._expectedTrainingImg = cv.imread(firstExpectedTrainingFileLocation)
        if self._expectedTrainingImg is None:
            self.logger.error('training sample file {} deleted?'
                              'please set --skipCapture 0 and rerun'.
                              format(firstExpectedTrainingFileLocation))
            raise ValueError

    def _makeJudgement(self):
        """
         load all trained training images and compare each image with currently captured live image
         if currently captured frame are match or not, show result in windows
        :return:
        """
        while self._windowManager.isWindowCreated:
            # reset serial port buffer, discard garbage data once every time operator changes LCD module
            self._communicationManager.resetInputBuffer()
            self._communicationManager.resetOutputBuffer()
            for self._current in self._predefinedPatterns:
                if not self._windowManager.isWindowCreated:  # user pressed ESC ,trying to quit program
                    if self._captureManager.cameraIsOpened:
                        self._captureManager.closeCamera()
                    if self._communicationManager is not None:
                        self._communicationManager.close()
                    self.logger.info('user pressed ESC to quit.')
                    break

                if self._current == SKIP_STATE_ID:
                    self.logger.debug('skip capturing training sample of id = {}'.format(SKIP_STATE_ID))
                    continue
                response = b''
                retry = 0
                while response == b'' and retry < 3:
                    self.logger.debug('send command to switch to {},retry={}'.format(self._current, retry))
                    assert self._current in range(1, STATES_NUM+1)
                    command = self._communicationManager.send(self._current)
                    response = self._communicationManager.getResponse()
                    if response[:-1] == command[:]:
                        self.logger.info('===>>> get valid response, wait for {} second,\n'
                                         ' then compare'
                                         ' positive training images for type {}'.format(DELAY_IN_SECONDS,self._current))
                        time.sleep(DELAY_IN_SECONDS)
                        # load the expected training sample
                        firstExpectedTrainingFileLocation = self._path(self._current, self.positive, 0)
                        self._expectedTrainingImg = cv.imread(firstExpectedTrainingFileLocation)
                        if self._expectedTrainingImg is None:
                            self.logger.error('training sample file {} deleted?'
                                              'please set --skipCapture 0 and rerun'.
                                              format(firstExpectedTrainingFileLocation))
                            raise ValueError
                        if not self._captureManager.cameraIsOpened:
                            self._captureManager.openCamera()
                        self._waitFrameCount = 0
                        while self._windowManager.isWindowCreated and self._waitFrameCount <= \
                                self._trainingFrameCount:
                            self._captureManager.enterFrame()
                            if self._waitFrameCount == self._trainingFrameCount:
                                # enable smooth mode, suppress noise
                                self._captureManager.setCompareModel(self._current, self.interestedMask,
                                                                     self._expectedTrainingImg, True)
                            compareResult = self._captureManager.exitFrame()
                            if compareResult is not None:
                                if compareResult['matched']:
                                    self.__testResults.append(1)
                                else:
                                    self.__testResults.append(0)
                                    self.logger.warn('NOT MATCH FOUND for expectedId {}'.format(self._current))
                            self._waitFrameCount += 1
                            self._windowManager.processEvents()
                        if self._captureManager.cameraIsOpened:
                            self._captureManager.closeCamera()

                    elif response == b'':
                        self.logger.debug('no response after few seconds timeout')
                        retry += 1
                    else:
                        self.logger.error("Receive a mismatched response from controller,skip this pic :{}".format(response))
                        self.__testResults.append(0)
                        break

                if retry == 3:
                    self.logger.debug('pic_id:{},communication timeout,no response from controller after 3 times retry!'.format(self._current))
                    self.__testResults.append('N')
                    continue
            self.reportTestResult()

        if self._captureManager.cameraIsOpened:
            self._captureManager.closeCamera()
        if self._communicationManager is not None:
            self._communicationManager.close()
        self.logger.info("test ends per user's request")

        # self._captureManager.openCamera()
        # while self._windowManager.isWindowCreated:
        #     self._captureManager.enterFrame()
        #     # enable smooth mode, suppress noise
        #     self._captureManager.setCompareModel(self.expectedTrainingImageId, self.interestedMask,
        #                                          self._expectedTrainingImg, True)
        #     compareResult = self._captureManager.exitFrame()
        #     # if compareResult is not None and compareResult['matched']:
        #     #     # load the training sample specified inside compareResult
        #     #     trainingFileLocation = join(self._folderName,
        #     #                                      str(compareResult['predictedClassId']),
        #     #                                      self.positive + '0.' + self._imgFormat)
        #     #     self._expectedTrainingImg = cv.imread(trainingFileLocation)
        #     #     if self._expectedTrainingImg is None:
        #     #         self.logger.error('training sample file {} deleted?'
        #     #                   'please set --skipCapture 0 and rerun'.
        #     #                       format(trainingFileLocation))
        #     self._windowManager.processEvents()
        # self._captureManager.closeCamera()





    def reportTestResult(self):
        """
        report this LCD's overall test result, on windowManager screen,prompt operator to remove current LCD and install
        next LCD module
        :return:
        """
        # self._communicationManager.close()
        self.logger.info('result is: {}'.format(self.__testResults))
        self._readyToNext = False
        if not self._captureManager.cameraIsOpened:
            self._captureManager.openCamera()
        while self._windowManager.isWindowCreated and self._readyToNext is False:
            self._captureManager.enterFrame()
            # in response to all of those user key input commands which is set in self.onKeyPress method
            # support user to  switch DUT images and check each training images with current live captured images
            # this is useful for debugging.
            self._captureManager.setCompareModel(self.expectedTrainingImageId, self.interestedMask,
                                                 self._expectedTrainingImg, False)
            # report last result(Pass or fail), prompt user to switch LCD module
            prompt = 'change LCD module, when ready, press RETURN to continue'
            self._captureManager.setUserPrompt(self.__testResults, prompt)
            compareResult = self._captureManager.exitFrame()
            self._windowManager.processEvents()


    def onKeyPress(self,keyCode):
        """
        key handler to communicate with window
        :param keyCode:
        :return:
        """
        if keyCode == ord('q') or keyCode == 27:
            self._windowManager.destroyWindow()
            self._snapshotWindowManager.destroyWindow()
            self._captureManager.closeCamera()

        elif keyCode == 32:  # space key
            self.logger.info('save screenshoot to snapshot.png')
            self._captureManager.save('snapshot.png')
        elif keyCode in (ord(','), ord('.')): # simulate adding/decreasing expected SVM model/training image id
            if keyCode == ord(','):
                if self.expectedTrainingImageId > 1:
                    self.expectedTrainingImageId -= 1
                else:
                    self.expectedTrainingImageId = STATES_NUM
                if self.expectedTrainingImageId == SKIP_STATE_ID:
                    self.expectedTrainingImageId -= 1
            else:
                if self.expectedTrainingImageId < STATES_NUM:
                    self.expectedTrainingImageId += 1
                else:
                    self.expectedTrainingImageId = 1
                if self.expectedTrainingImageId == SKIP_STATE_ID:
                    self.expectedTrainingImageId += 1
            try:
                firstExpectedTrainingFileLocation = self._path(self.expectedTrainingImageId, self.positive, 0)
                self._expectedTrainingImg = cv.imread(firstExpectedTrainingFileLocation)
            except Exception as e:
                self.logger.error('training sample file {} deleted? {}'.
                                  format(firstExpectedTrainingFileLocation, e))
            assert self._expectedTrainingImg is not None
            self.logger.info('use training Sample id {} to judge next frame'.format(self.expectedTrainingImageId))
        elif keyCode == ord('n') or keyCode == ord('N'):  # simulate move DUT to next image,skip one
            self._onDisplayId += 1
            if self._onDisplayId == STATES_NUM+1:
                self._onDisplayId = 1
            elif self._onDisplayId == SKIP_STATE_ID:
                self._onDisplayId += 1
            command = self._communicationManager.send(self._onDisplayId)
            response = self._communicationManager.getResponse()
            if response[:-1] == command[:]:
                self.logger.info('===>>> get valid response from DUT,\nDUT moves to next image type {}'.format(self._onDisplayId))
        elif keyCode == ord('b') or keyCode == ord('B'):  # simulate move DUT to previous image ,skip one
            self._onDisplayId -= 1
            if self._onDisplayId < 1:
                self._onDisplayId = STATES_NUM
            elif self._onDisplayId == SKIP_STATE_ID:
                self._onDisplayId -= 1
            command = self._communicationManager.send(self._onDisplayId)
            response = self._communicationManager.getResponse()
            if response[:-1] == command[:]:
                self.logger.info('===>>> get valid response from DUT,\nDUT moves to previous image type {}'.format(self._onDisplayId))
        elif keyCode == ord('d') or keyCode == ord('D'):  #draw rectangle over interestedMask Area
            if self._captureManager.displayImageType in [0]:
                roi = self._roi_box
            elif self._captureManager.displayImageType in[1, 2]:
                roi = self._warpImgBox
            self._snapshotWindowManager.setRectCords(roi)
            self._windowManager.setRectCords(roi)
            self._snapshotWindowManager.setDrawRect(True)
            self._windowManager.setDrawRect(True)
            self.logger.info('draw rectangle/keypoints in region of interest')
        elif keyCode == ord('u') or keyCode == ord('U'):  #undraw rectangle over interestedMask Area
            self._snapshotWindowManager.setDrawRect(False)
            self._windowManager.setDrawRect(False)
            self.logger.info('undraw rectangle/keypoints in region of interest')
        elif keyCode == ord('o'):  #show origin image
            self._captureManager.setDisplayImageType(0)
            roi = self._roi_box
            self._snapshotWindowManager.setRectCords(roi)
            self._windowManager.setRectCords(roi)
            self.logger.info('show original image in window')
        elif keyCode == ord('p'):  #show processed image in live capture window
            self._captureManager.setDisplayImageType(1)
            roi = self._warpImgBox
            self._snapshotWindowManager.setRectCords(roi)
            self._windowManager.setRectCords(roi)
            self.logger.info('show image after preProcessing in  window')
        elif keyCode == ord('r'):  #show warped region of interest image  in live capture window
            self._captureManager.setDisplayImageType(2)
            roi = self._warpImgBox
            self._snapshotWindowManager.setRectCords(roi)
            self._windowManager.setRectCords(roi)
            self.logger.info('show warped image of interested region in window')
        elif keyCode == 0x0d:  # return key to start new round to test next LCD module
            self._readyToNext = True
            self.__testResults.clear()
            self._captureManager.setUserPrompt(self.__testResults, None)
        else:
            self.logger.info('unknown key {} pressed'.format(chr(keyCode)))

def durationChecker(dura):
    num = int(dura)

    if num < 1:
        raise argparse.ArgumentTypeError(' minimum allowable value is 1')
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="standard training image sets build up and predict")
    parser.add_argument("--device", dest='deviceId', help='video camera ID [0,1,2,3]', default=0, type=int)
    parser.add_argument("--port", dest='portId', help='USB serial com port ID [0,1,2,3]', default=0, type=int)
    parser.add_argument("--width", dest='width', help='set video camera resolution width [1920,1280,800,640,160 etc]',
                        default=1920, type=int)
    parser.add_argument("--height", dest='height', help='set video camera resolution height [1080,960,600,480,120 etc]',
                        default=1080, type=int)
    parser.add_argument("--duration", dest='duration', help='how many samples to capture as training samples[>=1]',
                        default=1, type=durationChecker)
    parser.add_argument("--imgWidth", dest='imgWidth', help='set ROI image width [800, 600,300,etc]',
                        default=600, type=int)
    parser.add_argument("--imgHeight", dest='imgHeight', help='set ROI image height [800, 600,300,etc]',
                        default=600, type=int)
    parser.add_argument("--folder", dest='folder', help='folder name to store training image files',
                        default="/media/newdiskp1/picMatch/trainingImages", type=str)
    parser.add_argument("--imageFormat", dest='imageFormat', help='image format [png,jpg,gif,jpeg]',
                        default='png', type=str)
    parser.add_argument("--skipCapture", dest='skipCapture', help='do not overwrite existing image files [0,1]',
                        default=True, type=int)
    parser.add_argument("--blurValue", dest='blurValue', help='user defined blur level[1,255]', default=9, type=int)
    parser.add_argument("--cameraNoise", dest='cameraNoise', help='user defined camera noise level [0,255]', default=8,
                        type=int)
    parser.add_argument("--imageTheme", dest='imageTheme', help='user defined training images theme [0,3]', default=1,
                        type=int)
    parser.add_argument("--ssThreshold", dest='ssThreshold',
                        help='user defined structure similarity threshold[0,255]\n default=23', default=23, type=int)
    parser.add_argument("--cameraOffsetX", dest='offsetX',
                        help='allowable camera shift in pixel in X direction away from training position[0,255]\n '
                             'default=3', default=3, type=int)
    parser.add_argument("--cameraOffsetY", dest='offsetY',
                        help='allowable camera shift in pixel in Y direction away from training position[0,255]\n '
                             'default=3', default=3, type=int)
    parser.add_argument("--deltaArea", dest='deltaArea', help=' maximum area difference [0,50000],default=2000',
                        default=2000, type=int)
    parser.add_argument("--deltaCenterX", dest='deltaCenterX',
                        help='maximum center coordination difference in X [0,255],default=30', default=30, type=int)
    parser.add_argument("--deltaCenterY", dest='deltaCenterY',
                        help='maximum center coordination difference in Y [0,255],default=30', default=30, type=int)
    parser.add_argument("--deltaRadius", dest='deltaRadius',
                        help='maximum radius difference in pixel [0,255],default=30', default=30, type=int)
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s %(message)s')
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logHandler)
    logger.info(args)
    # if args.thresholdOffset is None:
    #     logger.info('please specify a mandatory CRITICAL training parameter thresholdOffset '
    #                 'using --thresholdOffset t, range is [0,255],\n'
    #                 'The larger the value, more symbol part on LCD display will show up, '
    #                 'threshTuneTrackbar utility can be used to search for it!')
    #     exit(-1)
    solution = BuildDatabase(logger, "live capture window", args.deviceId,
                             range(1, STATES_NUM+1, 1), args.portId, args.duration,
                             args.width, args.height, args.imgWidth,args.imgHeight, args.folder,
                             imgFormat=args.imageFormat,
                             skipCapture=args.skipCapture,
                             blurLevel=args.blurValue,
                             noiseLevel = args.cameraNoise, imageTheme=args.imageTheme,
                             structureSimilarityThreshold=args.ssThreshold,
                             offsetX=args.offsetX,
                             offsetY=args.offsetY,
                             deltaArea=args.deltaArea,
                             deltaCenterX=args.deltaCenterX,
                             deltaCenterY=args.deltaCenterY,
                             deltaRadius=args.deltaRadius)
    try:
        solution.run()
        # solution.makeJudgement()
    except ValueError as e:
        logger.error(e)
        pass
    # solution.reportTestResult()




