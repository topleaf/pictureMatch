
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

DELAY_IN_SECONDS = 1



# scale = 2
# wP = 300*scale
# hP = 300*scale

SX, SY = 542, 103
EX, EY = 1063, 687
RU_X, RU_Y = 1072, 114
LB_X, LB_Y = 529, 675
#
# SX, SY = 911, 87
# EX, EY = 1500, 756
# RU_X, RU_Y = 1554, 140
# LB_X, LB_Y = 900, 732

class BuildDatabase(object):
    # discard how many frames before taking a snapshot for comparison
    def __init__(self,logger,windowName,captureDeviceId,predefinedPatterns,portId,
                 duration, videoWidth, videoHeight, wP, hP, folderName,roiFolderName,featureFolder,
                 imgFormat,modelFolder,modelPrefixName,skipCapture=True,reTrainModel=True,
                 thresholdValue=47, blurLevel=9,noiseLevel=8,imageTheme=3,structureSimilarityThreshold=23,
                 offsetX=5,offsetY=5,deltaArea=40,deltaCenterX=20,deltaCenterY=20,deltaRadius=10):
        """

        :param windowName: the title of window to show captured picture,str
        :param captureDeviceId:  the device id of the camera
        :param predefinedPatterns:
        :param portId: serial port id: int
        """
        #define roi mask coordinations in WarpImage
        self._S_MASKX, self._E_MASKX, self._S_MASKY, self._E_MASKY = 5, wP-5, 5, hP-5
        self.logger = logger
        self._compareResultList = []
        self._roi_box = [(SX, SY), (LB_X, LB_Y), (EX, EY), (RU_X, RU_Y)]

        # normalize coordinates to integers
        self.box = np.int0(self._roi_box)
        self._snapshotWindowManager = WindowManager("Training Sample", None)
        self._windowManager = WindowManager(windowName, self.onKeyPress)

        self._captureManager = CaptureManager(logger, captureDeviceId,
                                              self._windowManager, self._snapshotWindowManager,
                                              False, videoWidth, videoHeight,
                                              self._compareResultList,
                                              warpImgSize=(wP, hP),
                                              threshValue=thresholdValue,
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
                                                              self._expireSeconds,algorithm=imageTheme)
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

        self._skipCapture = skipCapture
        self.positive = 'pos-'      # positive training samples filename prefix
        self._modelFolder = modelFolder
        self._modelPrefixName = modelPrefixName
        self._bowExtractorListFilename = 'bowExtractorList'
        self._maskFileName = 'maskZone'
        # self.interestedMask = None      # define a region of interest after get image from frame
        # set up a mask to select interested zone only
        self.interestedMask = np.zeros((self._warpImgHP, self._warpImgWP), np.uint8)
        self.interestedMask[self._S_MASKY:self._E_MASKY, self._S_MASKX:self._E_MASKX] = np.uint8(255)
        self.BOW_CLUSTER_NUM = STATES_NUM-1  # bag of visual words cluster number
        self.expectedSvmModelId = 1  # designated SVM model id or expected training sample id to be compared with current image
        self.svmModels = []
        self.vocs = []
        self.extractBowList = []
        self._onDisplayId = STATES_NUM
        self._retrainModel = reTrainModel
        self._drawOrigin = False   # draw origin snapshot or processed snapshot in live capture window ?
        self._expectedTrainingImg = None  # load expected svmmodel's first training image
        self._startTime = time.time()
        #create  keypoints detector and descriptor extractor
        self.detector = cv.xfeatures2d.SIFT_create(nOctaveLayers=5)    # the larger nOctaveLayer, the smaller blob it can detect
        self.extract = cv.xfeatures2d.SIFT_create(nOctaveLayers=5)
        #create a flann matcher
        FLANN_INDEX_KDTREE = 1 # kd tree
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        searchParams = {} #dict(checks=50)
        self.flann = cv.FlannBasedMatcher(indexParams, searchParams)
        # create a bowImgDescriptorExtractor
        self.extractBow = cv.BOWImgDescriptorExtractor(self.extract, self.flann)
        #create a bag-of-word K means trainer
        self.bowKmeansTrainer = cv.BOWKMeansTrainer(self.BOW_CLUSTER_NUM)




    def run(self):

        #  step 1: capture and save all positive training images to respective image folders/pos-x
        self._captureAndSaveAllImages()
        if not self._retrainModel:
            return
        #  step 2: training each SVM models and save them to disk.
        # self._trainSVMModels()

        # or train one multi-classification SVM mode and save to disk
        # self._trainSVMModel()
        training_duration = time.time() - self._startTime

        self.logger.info('Training duration is:{} seconds'.format(training_duration))



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

    def _path(self,typeid,cls,i):
        '''
        get positive or negative image files full path
        :param typeid:  positive image type id
        :param cls: 'pos-' or 'neg-'
        :param i:  sample id in int  , 0 < i < STATES_NUM
        :return:
        '''
        if cls == self.positive:
            return "%s/%s%d.%s" %(join(self._folderName,str(typeid)),cls,i,self._imgFormat)
        else:  # no neg- folder
            raise  ValueError


    def _captureAndSaveAllImages(self):
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
                                self.__testResults.append(
                                    self._captureManager.save(join(newFolder, self.positive + str(self._waitFrameCount) +
                                                              '.' + self._imgFormat)))
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
        firstExpectedTrainingFileLocation = join(self._folderName,
                                                 str(self.expectedSvmModelId),
                                                 self.positive +'0.' + self._imgFormat)
        self._expectedTrainingImg = cv.imread(firstExpectedTrainingFileLocation)
        if self._expectedTrainingImg is None:
            self.logger.error('training sample file {} deleted?'
                              'please set --skipCapture 0 and rerun'.
                              format(firstExpectedTrainingFileLocation))
            raise ValueError

    def get_keypoints(self, fn):
        """

        :param fn: file location
        :return: imgWarp and keypoints that lies in file within self.interestedMask region
        """
        im = cv.imread(fn, cv.IMREAD_UNCHANGED)
        # remove noise introduced by camera, pixel dance
        blur = self._captureManager.preProcess(im)

        # warp the interested ROI
        imgWarp = warpImg(blur, self.box, self._warpImgWP, self._warpImgHP)
        # use previously-setup mask to select the same interested zone only
        keypoints = self.detector.detect(imgWarp, mask=self.interestedMask)
        return imgWarp, keypoints

    def extract_sift(self,fn):
        """
        get descriptor of a image in a give path
        :param fn: full path of the image
        :return:
        """
        imgWarp, keypoints = self.get_keypoints(fn)
        keypoints, features = self.extract.compute(imgWarp, keypoints)
        return features


    def _bowFeatures(self,fn):
        """
        get descriptor of a given path image
        :param fn:  full path of the image
        :return:  its descriptor
        """
        imgWarp, keypoints = self.get_keypoints(fn)
        features = self.extractBow.compute(imgWarp, keypoints)  # based on vocabulary in self.extractBow
        return features

    def _trainSVMModel(self):
        """
        train one multiple classification SVM model for all STATE_NUM classes
        :return:
        """

        try:
            # insert all classes'  all training images' SIFT feature into bowKmeansTrainer
            for i in range(1, STATES_NUM+1, 1):
                if i != SKIP_STATE_ID:
                    self.logger.info('add training samples of class {}  to bowKmeansTrainer'.format(i))
                    for j in range(self._trainingFrameCount):
                        fileLocation = self._path(i, self.positive, j)
                        self.logger.debug('add training class {} sample {} to bowKmeansTrainer'.format(i,
                            fileLocation))
                        try:
                            features = self.extract_sift(fileLocation)
                        except Exception as e:
                            self.logger.error('{},training class {} sample {} for bowKmeansTrainer'
                                              ' has no SIFT features'.format(e, i, fileLocation))
                            raise ValueError
                            return
                        if features is not None:
                            self.bowKmeansTrainer.add(features)
                        else:
                            self.logger.error('training class {} sample {} for bowKmeansTrainer'
                                              ' has no SIFT features,\n'
                                              'Please adjust threhold  and recapture training images'.format(i, fileLocation))
                            raise ValueError
                            return

        except Exception as e:
            self.logger.error(e)
            raise ValueError
            return

        # cluster features descriptors from all training images into BOW_CLUSTER_NUM caterogies, it will
        # serve as vocabulary
        self.logger.info('start bowKmeansTrainer cluster calculation')
        voc = self.bowKmeansTrainer.cluster()
        self.extractBow.setVocabulary(voc)
        self.logger.info('bowKmeansTrainer cluster completes')

        # remember voc as well as self.extractBow to list for predicting use later
        self.vocs.append(voc)
        self.extractBowList.append(self.extractBow)

        # create all classes' positive training samples and marked them with relative labels
        # self._bowFeatures function will compute bowFeatures based on voc
        trainData, trainLabels = [], []
        for classId in range(1, STATES_NUM+1,1):
            if classId != SKIP_STATE_ID:
                self.logger.info('put positive training samples of class {}  to SVM model'.format(classId))
                for i in range(self._trainingFrameCount):
                    trainfileLocation = self._path(classId,self.positive, i)
                    self.logger.debug('put class {}  training bowFeature sample extracted from {} to SVM model'.format(
                        classId, trainfileLocation
                    ))
                    trainData.extend(self._bowFeatures(trainfileLocation))
                    trainLabels.append(classId)

        # create a svm and feed data to train the model
        self.logger.info('create unique {}-classification SVM model and train it ....'.format(STATES_NUM))
        svm = cv.ml.SVM_create()
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setGamma(0.5)
        svm.setC(30)  # trial
        svm.setKernel(cv.ml.SVM_RBF)
        svm.train(np.array(trainData), cv.ml.ROW_SAMPLE, np.array(trainLabels))
        self.logger.info('{}-classification SVM model training completes!'.format(STATES_NUM))
        try:
            mkdir(self._modelFolder)
        except FileExistsError:
            self.logger.debug('directory {} exists'.format(self._modelFolder))
            pass

        # save unique multiclass svm as index 0
        svm.save(join(self._modelFolder, self._modelPrefixName + str(0)))
        self.svmModels.append(svm)

        # save all of them to disk for future reload use if self._retrainModel is False
        try:
            np.save(join(self._modelFolder, self._bowExtractorListFilename), self.vocs)

            np.save(join(self._modelFolder, self._maskFileName), self.interestedMask)
        except Exception as e:
            self.logger.error(e)
            raise ValueError


    def _trainSVMModels(self):
        # training multiple svm models and save them to disk
        for self._current in self._predefinedPatterns:
            if self._current == SKIP_STATE_ID:
                self.logger.debug('skip training sample id = {}'.format(SKIP_STATE_ID))
                continue

            try:
                # insert the other typeid's first positive training image
                # as the negative training samples for this typeid to bowKmeansTrainer
                for i in range(1, STATES_NUM+1, 1):
                    if i != self._current and i != SKIP_STATE_ID:
                        fileLocation = self._path(i, self.positive, 1)
                        self.logger.debug('add negative training sample {} to model {} bowKmeansTrainer'.format(
                            fileLocation, self._current
                        ))
                        try:
                            features = self.extract_sift(fileLocation)
                        except Exception as e:
                            self.logger.error('{},negative training sample {} for model {} bowKmeansTrainer'
                                                ' has no SIFT features'.format(e, fileLocation, self._current))
                            raise ValueError
                            return
                        if features is not None:
                            self.bowKmeansTrainer.add(features)

                #insert positive training samples to bowKmeansTrainer   # which might be more than typeid
                for i in range(self._trainingFrameCount):
                    fileLocation = self._path(self._current, self.positive, i)
                    self.logger.debug('add positive training sample {} to model {} bowKmeansTrainer'.format(
                        fileLocation, self._current
                    ))
                    try:
                        features = self.extract_sift(fileLocation)
                    except Exception as e:
                        self.logger.error('{},positive training sample {} for model {} bowKmeansTrainer '
                                            'has no SIFT features'.format(e, fileLocation, self._current))
                        raise ValueError

                    if features is not None:
                        self.bowKmeansTrainer.add(features)


            except Exception as e:
                self.logger.error(e)
                raise ValueError
                return

            # cluster features descriptors from all training images into BOW_CLUSTER_NUM caterogies, it will
            # serve as vocabulary
            voc = self.bowKmeansTrainer.cluster()
            self.extractBow.setVocabulary(voc)

            # remember voc as well as self.extractBow to list for predicting use later
            self.vocs.append(voc)
            self.extractBowList.append(self.extractBow)

            # create both positive and negative training samples lists
            # self._bowFeatures function will compute bowFeatures based on voc
            trainData, trainLabels = [], []
            for i in range(self._trainingFrameCount):
                trainfileLocation = self._path(self._current,self.positive, i)
                self.logger.debug('put positive training bowFeature sample extracted from {} to SVM model {} '.format(
                    trainfileLocation, self._current
                ))
                trainData.extend(self._bowFeatures(trainfileLocation))
                trainLabels.append(1)

            for i in range(1, STATES_NUM+1, 1):
                if self._current != i and i != SKIP_STATE_ID:
                    trainfileLocation = self._path(i, self.positive, 1)
                    self.logger.debug('put negative training bowFeature sample extracted from {} to SVM model {} '.format(
                        trainfileLocation, self._current
                    ))
                    trainData.extend(self._bowFeatures(trainfileLocation))
                    trainLabels.append(-1)

            # create a svm and feed data to train the model
            print('create SVM model {} and train it ....'.format(self._current))
            svm = cv.ml.SVM_create()
            svm.train(np.array(trainData), cv.ml.ROW_SAMPLE, np.array(trainLabels))
            print('SVM model {} training completes! '.format(self._current))
            try:
                mkdir(self._modelFolder)
            except FileExistsError:
                self.logger.debug('directory {} exists'.format(self._modelFolder))
                pass
            svm.save(join(self._modelFolder, self._modelPrefixName + str(self._current)))
            self.svmModels.append(svm)

        # save all of them to disk for future reload use if self._retrainModel is False
        try:
            np.save(join(self._modelFolder, self._bowExtractorListFilename), self.vocs)
            np.save(join(self._modelFolder, self._maskFileName), self.interestedMask)
        except Exception as e:
            self.logger.error(e)
            raise ValueError



    def makeJudgement(self):
        """
         load all trained SVM models and apply expected svm model to predict
         if currently captured frame are match or not, show result in windows
        :return:
        """
        if not self._retrainModel:  # reload svm models/extractBowList/interestedMask from disk
            for rootdir,dirname,files in walk(self._modelFolder):
                files = sorted(files)
                self.svmModels.clear()
                for file in files:
                    if self._modelPrefixName in file:
                        self.svmModels.append(cv.ml.SVM_load(join(self._modelFolder, file)))
                        self.logger.info('reload SVM model:{}'.format(file))
                    elif self._bowExtractorListFilename in file:
                        self.extractBowList.clear()
                        try:
                            vocs = np.load(join(self._modelFolder, file))
                        except Exception as e:
                            self.logger.error(e)
                            raise ValueError
                        for voc in vocs:
                            self.extractBow.setVocabulary(voc)
                            self.extractBowList.append(self.extractBow)
                        self.logger.info('reload vocs file from {} and reconstruct extractBowList '.format(file))
                    elif self._maskFileName in file:
                        try:
                            self.interestedMask = np.load(join(self._modelFolder, file))
                        except Exception as e:
                            self.logger.error('failed to load maskfile:{}'.format(e))
                            raise ValueError
                        self.logger.info('reload interestedMask from {}'.format(file))

        self._captureManager.openCamera()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            # enable smooth mode, suppress noise
            self._captureManager.setCompareModel(self.expectedSvmModelId, self.svmModels,
                                                 self.extractBowList, self.interestedMask,
                                                 self._expectedTrainingImg, True)
            compareResult = self._captureManager.exitFrame()
            # if compareResult is not None and compareResult['matched']:
            #     # load the training sample specified inside compareResult
            #     trainingFileLocation = join(self._folderName,
            #                                      str(compareResult['predictedClassId']),
            #                                      self.positive + '0.' + self._imgFormat)
            #     self._expectedTrainingImg = cv.imread(trainingFileLocation)
            #     if self._expectedTrainingImg is None:
            #         self.logger.error('training sample file {} deleted?'
            #                   'please set --skipCapture 0 and rerun'.
            #                       format(trainingFileLocation))
            self._windowManager.processEvents()
        self._captureManager.closeCamera()





    def reportTestResult(self):
        self._communicationManager.close()
        self.logger.info('result is: {}'.format(self.__testResults))


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
                if self.expectedSvmModelId > 1:
                    self.expectedSvmModelId -= 1
                else:
                    self.expectedSvmModelId = STATES_NUM
                if self.expectedSvmModelId == SKIP_STATE_ID:
                    self.expectedSvmModelId -= 1
            else:
                if self.expectedSvmModelId < STATES_NUM:
                    self.expectedSvmModelId += 1
                else:
                    self.expectedSvmModelId = 1
                if self.expectedSvmModelId == SKIP_STATE_ID:
                    self.expectedSvmModelId += 1
            try:
                firstExpectedTrainingFileLocation = join(self._folderName,
                                                         str(self.expectedSvmModelId),
                                                         self.positive +'0.' + self._imgFormat)
                self._expectedTrainingImg = cv.imread(firstExpectedTrainingFileLocation)
            except Exception as e:
                self.logger.error('training sample file {} deleted? {}'.
                                  format(firstExpectedTrainingFileLocation, e))
            assert self._expectedTrainingImg is not None
            self.logger.info('use SVM model/training Sample id {} to judge next frame'.format(self.expectedSvmModelId))
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
            # if self._captureManager.displayImageType in [0, 1]:
            #     roi = self._roi_box
            # elif self._captureManager.displayImageType == 2:
            #     roi = self._warpImgBox
            # self._snapshotWindowManager.setRectCords(roi)
            # self._windowManager.setRectCords(roi)
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
        else:
            self.logger.info('unknown key {} pressed'.format(chr(keyCode)))

def durationChecker(dura):
    num = int(dura)

    if num < 2:
        raise argparse.ArgumentTypeError(' minimum allowable value is 2')
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="standard training image sets build up and predict")
    parser.add_argument("--device", dest='deviceId', help='video camera ID [0,1,2,3]',default=0, type=int)
    parser.add_argument("--port", dest='portId', help='USB serial com port ID [0,1,2,3]',default=0, type=int)
    parser.add_argument("--width", dest='width', help='set video camera width [1280,800,640,160 etc]',default=1920, type=int)
    parser.add_argument("--height", dest='height', help='set video camera height [960,600,480,120 etc]',default=1080, type=int)
    parser.add_argument("--duration", dest='duration', help='how many samples to capture as training samples[>=2]',
                        default=2, type=durationChecker)
    parser.add_argument("--imgWidth", dest='imgWidth', help='set ROI image width [1280,800,640,etc]',default=600, type=int)
    parser.add_argument("--imgHeight", dest='imgHeight', help='set ROI image height [960,600,480,etc]',default=600, type=int)
    parser.add_argument("--folder", dest='folder', help='folder name to store training image files', default= "/media/newdiskp1/picMatch/trainingImages", type=str)
    parser.add_argument("--roiFolder", dest='roiFolder', help='folder name to store Region of Interest training image files', default ='/media/newdiskp1/picMatch/rois', type=str)
    parser.add_argument("--featureFolder", dest='featureFolder', help='folder name to store image feature files', default ='/media/newdiskp1/picMatch/features', type=str)
    parser.add_argument("--imageFormat", dest='imageFormat', help='image format [png,jpg,gif,jpeg]', default ='png', type=str)
    parser.add_argument("--skipCapture", dest='skipCapture', help='do not overwrite existing image files [0,1]', default = True, type=int)
    parser.add_argument("--modelFolder", dest='modelFolder', help='folder name to store trained SVM models', default = '/media/newdiskp1/picMatch/models', type=str)
    parser.add_argument("--reTrain", dest='reTrain', help='retrain SVM models or NOT [0,1]', default = False, type=int)
    parser.add_argument("--threshold", dest='threshold', help='threshold value[1,255]',  type=int)
    parser.add_argument("--blurValue", dest='blurValue', help='user defined blur level[1,255]', default=9, type=int)
    parser.add_argument("--cameraNoise", dest='cameraNoise', help='user defined camera noise level [0,255]', default=8, type=int)
    parser.add_argument("--imageTheme", dest='imageTheme', help='user defined images theme [0,3]', default=1, type=int)
    parser.add_argument("--ssThreshold", dest='ssThreshold', help='user defined structure similarity threshold[0,255]\n default=23', default=23, type=int)
    parser.add_argument("--cameraOffsetX", dest='offsetX', help='allowable camera shift in pixel in X direction away from training position[0,255]\n default=5', default=5, type=int)
    parser.add_argument("--cameraOffsetY", dest='offsetY', help='allowable camera shift in pixel in Y direction away from training position[0,255]\n default=5', default=5, type=int)
    parser.add_argument("--deltaArea", dest='deltaArea', help=' maximum area difference [0,50000],default=2000', default=2000, type=int)
    parser.add_argument("--deltaCenterX", dest='deltaCenterX', help='maximum center coordination difference in X [0,255],default=20', default=20, type=int)
    parser.add_argument("--deltaCenterY", dest='deltaCenterY', help='maximum center coordination difference in Y [0,255],default=20', default=20, type=int)
    parser.add_argument("--deltaRadius", dest='deltaRadius', help='maximum radius difference in pixel [0,255],default=20', default=20, type=int)

    args = parser.parse_args()


    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s %(message)s')
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logHandler)

    logger.info(args)

    if args.threshold is None:
        logger.info('please specify a mandatory CRITICAL training parameter threshold '
                    'using --threshold t, range is [1,255],\nYou can use threshTuneTrackbar utility to find it!')
        exit(-1)
    solution = BuildDatabase(logger, "live capture window", args.deviceId,
                             range(1, STATES_NUM+1, 1), args.portId, args.duration,
                             args.width, args.height, args.imgWidth,args.imgHeight, args.folder, args.roiFolder,
                             args.featureFolder, imgFormat=args.imageFormat,
                             modelFolder=args.modelFolder, modelPrefixName='svmxml',skipCapture=args.skipCapture,
                             reTrainModel = args.reTrain, thresholdValue=args.threshold, blurLevel=args.blurValue,
                             noiseLevel = args.cameraNoise,imageTheme=args.imageTheme,
                             structureSimilarityThreshold=args.ssThreshold,
                             offsetX=args.offsetX,
                             offsetY=args.offsetY,
                             deltaArea=args.deltaArea,
                             deltaCenterX=args.deltaCenterX,
                             deltaCenterY=args.deltaCenterY,
                             deltaRadius=args.deltaRadius)
    try:
        solution.run()
        solution.makeJudgement()
    except ValueError as e:
        logger.error(e)
        pass
    solution.reportTestResult()




