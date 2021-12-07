
"""
Author: Jin Li
Date: Dec 2,2021
build query images database information,
1. capture all possible standard images from camera, save to disk
2. detect ROI (by corner , or by edgeDetection), warp image to 600*600 coordination system, save to disk
3. detect and compute their keypoints and featureDescriptor, save to disk
"""
from managers import WindowManager,CaptureManager,CommunicationManager
from edgeDetect import extractValidROI
import logging
import argparse
from os import walk, mkdir
import cv2 as cv
from cv2 import xfeatures2d
import numpy as np

class BuildDatabase(object):
    # discard how many frames before taking a snapshot for comparison
    def __init__(self,logger,windowName,captureDeviceId,predefinedPatterns,portId,
                 duration, videoWidth, videoHeight, wP, hP,folderName,roiFolderName,featureFolder,
                 imgFormat,skipCapture=True):
        """

        :param windowName: the title of window to show captured picture,str
        :param captureDeviceId:  the device id of the camera
        :param predefinedPatterns:
        :param portId: serial port id: int
        """

        self.logger = logger
        self._compareResultList = []
        self._snapshotWindowManager = WindowManager("snapshot Window", None)
        self._windowManager = WindowManager(windowName, self.onKeyPress)

        self._captureManager = CaptureManager(logger, captureDeviceId,
                                              self._windowManager, self._snapshotWindowManager,
                                              False, videoWidth, videoHeight,
                                              self._compareResultList,
                                              )
        self._predefinedPatterns = predefinedPatterns
        self._expireSeconds = 5
        self._discardFrameCount = duration
        try:
            self._communicationManager = CommunicationManager(self.logger, '/dev/ttyUSB'+str(portId),
                                                              self._expireSeconds)
        except ConnectionError:
            self.logger.error('Abort!! Please make sure serial port is ready then retry')
            exit(-1)
        # self.__capturedPicture = None
        self._windowManager.createWindow()
        self._snapshotWindowManager.createWindow()
        self.__testResults = []
        self._current = None
        self._waitFrameCount = 0  #
        self._folderName = folderName
        self._roiFolderName = roiFolderName
        self._featureFolderName = featureFolder
        self._imgFormat = imgFormat
        self._warpImgWP, self._warpImgHP = wP, hP
        self._skipCapture = skipCapture


    def run(self):
        #  step 1: capture and save all images one by one to image folders
        self._captureAndSaveAllImages()
        #  step 2: Detect ROI, generating and project to warp images, save to image folders
        #  generate image's keypoints&feature descriptors,save to featureFolders
        self._generateWarpImagesAndDescriptors()

    def _generateWarpImagesAndDescriptors(self):
        """
        from raw captured training pictures folder, generate training pictures to be used as standard samples
        save the samples and their respective graph feature Descriptors to self._roiFolderName and self._featureFolderName
        :return:
        """
        try:
            mkdir(self._roiFolderName)
        except FileExistsError as e:
            self.logger.debug('roi image folder %s exists' % self._roiFolderName)

        files = []
        for (dirPath, dirNames, fileNames) in walk(self._folderName):
            pass
            # files.extend(fileNames)
        for file in fileNames:      # only keep files with required extension name
            if file.split('.')[1] in [self._imgFormat]:
                files.append(file)

        for fileName in files:
            fullPath = self._folderName + '/' + fileName
            rawImg = cv.imread(fullPath)
            retval, warpImg = extractValidROI(rawImg, drawRect=False, save=False,
                                              wP=self._warpImgWP, hP=self._warpImgHP, display=False)

            if retval:
                warpFileName = self._roiFolderName + '/' + fileName.split('.')[0]+'_ROI.' + self._imgFormat
                cv.imwrite(warpFileName, warpImg)
                self.logger.info('ROI detected, saved to {}'.format(warpFileName))

                siftFeatureDetector = cv.xfeatures2d.SIFT_create()
                kp, featureDescriptor = siftFeatureDetector.detectAndCompute(warpImg, None)
                descriptorFileName = self._featureFolderName + '/' + \
                                     fileName.replace(self._imgFormat, 'npy')
                kpFileName = self._featureFolderName + '/' +\
                             fileName.replace(self._imgFormat, 'kpy')
                np.save(descriptorFileName, featureDescriptor)
                # np.save(kpFileName, kp)

                self.logger.info('keypoints and descriptors saved to {} {}'.format(
                    kpFileName, descriptorFileName
                ))
            else:
                self.logger.error('ROI not detected in file {}, please adjust camera or tune parameters '
                                  'and retry again!!'.format(fullPath))
                # raise ValueError



    def _captureAndSaveAllImages(self):
        try:
            mkdir(self._folderName)
        except FileExistsError as e:
            self.logger.debug('image folder %s exists' %self._folderName)
            if self._skipCapture:
                self.logger.debug('skip overwriting')
                return
        try:
            mkdir(self._featureFolderName)
        except FileExistsError as e:
            self.logger.debug('feature folder %s exists' %self._featureFolderName)

        for self._current in self._predefinedPatterns:
            response = b''
            retry = 0
            while response == b'' and retry < 3:
                self.logger.debug('send command to switch to {},retry={}'.format(self._current, retry))
                assert self._current in range(STATES_NUM)
                command = self._communicationManager.send(self._current)
                response = self._communicationManager.getResponse()
                if response[:-1] == command:
                    self.logger.info('===>>> get valid response, start  camera capture and preview')

                    self._waitFrameCount = 0
                    while self._windowManager.isWindowCreated and self._waitFrameCount <= \
                            self._discardFrameCount:
                        self._captureManager.enterFrame()
                        if self._waitFrameCount == self._discardFrameCount:
                            # capture and save the next frame img to file
                            # in current directory by the name of self._current
                            self.__testResults.append(
                                self._captureManager.save(self._folderName+'/'
                                                          + str(self._current) +
                                                          '.' + self._imgFormat))
                        self._waitFrameCount += 1
                        self._captureManager.exitFrame()
                        self._windowManager.processEvents()

                elif response == b'':
                    self.logger.debug('no response after few seconds timeout')
                    retry += 1
                else:
                    self.logger.error("Receive a mismatched response from controller,skip this pic :{}".format(response))
                    self.__testResults.append('O')
                    self._waitFrameCount = True
                    break

            if retry == 3:
                self.logger.debug('pic_id:{},communication timeout,no response from controller after 3 times retry!'.format(self._current))
                self.__testResults.append('N')
                self._waitFrameCount = True
                continue


    def reportTestResult(self):
        self._communicationManager.close()
        self.logger.info('train completes:')
        self.logger.info('result is: {}'.format(self.__testResults))



    def onKeyPress(self,keyCode):
        if keyCode == ord('q') or keyCode == 27:
            self._windowManager.destroyWindow()
            self._snapshotWindowManager.destroyWindow()
        elif keyCode == 32:  # space key
            self.logger.info('save screenshoot to snapshot.png')
            self._captureManager.save('snapshot.png')
        elif keyCode == ord('n') or keyCode == ord('N'):
            self._waitFrameCount = True
        else:
            self.logger.debug('unknown key {} pressed'.format(chr(keyCode)))

        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="standard training image sets build up")
    parser.add_argument("--device", dest='deviceId', help='video camera ID [0,1,2,3]',default=0, type=int)
    parser.add_argument("--port", dest='portId', help='USB serial com port ID [0,1,2,3]',default=0, type=int)
    parser.add_argument("--width", dest='width', help='set video camera width [1280,800,640,160 etc]',default=800, type=int)
    parser.add_argument("--height", dest='height', help='set video camera height [960,600,480,120 etc]',default=600, type=int)
    parser.add_argument("--duration", dest='duration',help='how many frames to discard before confirmation',default=50, type=int)
    parser.add_argument("--imgWidth", dest='imgWidth', help='set ROI image width [1280,800,640,etc]',default=600, type=int)
    parser.add_argument("--imgHeight", dest='imgHeight', help='set ROI image height [960,600,480,etc]',default=600, type=int)
    parser.add_argument("--folder", dest='folder',help='folder name to store training image files', default= "./pictures", type=str)
    parser.add_argument("--roiFolder", dest='roiFolder',help='folder name to store Region of Interest training image files', default ='./rois', type=str)
    parser.add_argument("--featureFolder", dest='featureFolder',help='folder name to store image feature files', default ='./features', type=str)
    parser.add_argument("--imageFormat", dest='imageFormat',help='image format [png,jpg,gif,jpeg]', default ='png', type=str)
    parser.add_argument("--skipCapture", dest='skipCapture',help='do not overwrite existing image files [0,1]', default = True, type=int)

    args = parser.parse_args()

    STATES_NUM = 16
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s %(message)s')
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logHandler)

    logger.info(args)


    solution = BuildDatabase(logger, "build database", args.deviceId,
                             range(0, STATES_NUM, 1), args.portId, args.duration,
                             args.width, args.height, args.imgWidth,args.imgHeight, args.folder, args.roiFolder,
                             args.featureFolder, imgFormat=args.imageFormat,
                             skipCapture=args.skipCapture)
    try:
        solution.run()
    except ValueError:
        pass
    solution.reportTestResult()




