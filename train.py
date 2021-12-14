
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
"""
from managers import WindowManager,CaptureManager,CommunicationManager
from edgeDetect import extractValidROI
import logging
import argparse
from os import walk, mkdir
from os.path import join
import cv2 as cv
from cv2 import xfeatures2d
import numpy as np

STATES_NUM = 3
SY,EY = 358, 882
SX,EX = 1150, 1748

class BuildDatabase(object):
    # discard how many frames before taking a snapshot for comparison
    def __init__(self,logger,windowName,captureDeviceId,predefinedPatterns,portId,
                 duration, videoWidth, videoHeight, wP, hP,folderName,roiFolderName,featureFolder,
                 imgFormat,modelFolder,modelPrefixName,skipCapture=True):
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
        self._trainingFrameCount = duration
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
        self.positive = 'pos-'      # positive training samples filename prefix
        self.negative = 'neg-'
        self._modelFolder = modelFolder
        self._modelPrefixName = modelPrefixName
        self.interestedMask = None      # define a region of interest after get image from frame



    def run(self):
        #  step 1: capture and save all positive training images to respective image folders/pos-x
        self._captureAndSaveAllImages()
        #  step 2: training each SVM models and save them to disk.
        self._trainSVMModels()

        # step 3: predict current captured frame by different models to check accuracy
        self._predict()

        # #  generate image's keypoints&feature descriptors,save to featureFolders
        # self._generateWarpImagesAndDescriptors()

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
            self.logger.debug('image folder %s exists' %self._folderName)
            if self._skipCapture:
                self.logger.debug('skip overwriting')
                return


        for self._current in self._predefinedPatterns:
            response = b''
            retry = 0
            while response == b'' and retry < 3:
                self.logger.debug('send command to switch to {},retry={}'.format(self._current, retry))
                assert self._current in range(STATES_NUM)
                command = self._communicationManager.send(self._current)
                response = self._communicationManager.getResponse()
                if response[:-1] == command:
                    self.logger.info('===>>> get valid response, start capturing and saving'
                                     ' positive training images for type {}'.format(self._current))
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


    def extract_sift(self,fn):
        """
        get descriptor of a image in a give path
        :param fn: full path of the image
        :return:
        """

        im  = cv.imread(fn, 0)
        # set up a mask to select interested zone only
        self.interestedMask = np.zeros((im.shape[0], im.shape[1]), np.uint8)
        # self.interestedMask[0:im.shape[0], 0:im.shape[1]] = np.uint8(255)
        self.interestedMask[SY:EY, SX:EX] = np.uint8(255)
        # cv.imshow('mask', self.interestedMask)
        keypoints = self.detector.detect(im, mask=self.interestedMask)
        keypoints, features = self.extract.compute(im, keypoints)
        return features


    def _bowFeatures(self,fn):
        """
        get descriptor of a given path image
        :param fn:  full path of the image
        :return:  its descriptor
        """
        im = cv.imread(fn, 0)
        # # set up a mask to select interested zone only
        # self.interestedMask = np.zeros((im.shape[0], im.shape[1]), np.uint8)
        # # self.interestedMask[0:im.shape[0], 0:im.shape[1]] = np.uint8(255)
        # self.interestedMask[SY:EY, SX:EX] = np.uint8(255)
        keypoints = self.detector.detect(im, mask=self.interestedMask)
        features = self.extractBow.compute(im, keypoints)
        return features


    def _trainSVMModels(self):
        # training multiple svm models and save them to disk
        self.svmModels = []
        for self._current in self._predefinedPatterns:
            #create  keypoints detector and descriptor extractor
            self.detector = cv.xfeatures2d.SIFT_create()
            self.extract = cv.xfeatures2d.SIFT_create()

            #create a flann matcher
            FLANN_INDEX_KDTREE = 0
            indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            searchParams = dict(checks=50)
            self.flann = cv.FlannBasedMatcher(indexParams, searchParams)


            #create a bag-of-word K means trainer with 40 clusters
            self.bowKmeansTrainer = cv.BOWKMeansTrainer(40)

            # create a bowImgDescriptorExtractor
            self.extractBow = cv.BOWImgDescriptorExtractor(self.extract, self.flann)

            try:
                # insert the other typeid's first positive training image
                # as the negative training samples for this typeid to bowKmeansTrainer
                for i in range(1, STATES_NUM, 1):
                    if i != self._current:
                        self.bowKmeansTrainer.add(self.extract_sift(self._path(i, self.positive, 1)))

                #insert positive training samples to bowKmeansTrainer   # which might be more than typeid
                for i in range(self._trainingFrameCount):
                    self.bowKmeansTrainer.add(self.extract_sift(self._path(self._current,self.positive,i)))
            except Exception as e:
                self.logger.error(e)
                raise ValueError

            # generate vocabulary
            voc = self.bowKmeansTrainer.cluster()
            self.extractBow.setVocabulary(voc)

            # create both positive and negative training samples lists
            trainData, trainLabels = [], []
            for i in range(self._trainingFrameCount):
                trainData.extend(self._bowFeatures(self._path(self._current, self.positive, i)))
                trainLabels.append(1)

            for i in range(1, STATES_NUM, 1):
                if self._current != i:
                    trainData.extend(self._bowFeatures(self._path(i, self.positive, 1)))
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


    def _predict(self):
        """
         load all trained SVM models and predict current captured frame
        :return:
        """
        self._captureManager.enterFrame()
        currentImage = self._captureManager.frame
        graySnapshot = cv.cvtColor(currentImage, cv.COLOR_BGR2GRAY)
        self._captureManager.exitFrame()
        i = 1
        if graySnapshot is not None:
            for svm in self.svmModels:
                #create  keypoints detector and descriptor extractor
                self.detector = cv.xfeatures2d.SIFT_create()
                self.extract = cv.xfeatures2d.SIFT_create()

                #create a flann matcher
                FLANN_INDEX_KDTREE = 0
                indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                searchParams = dict(checks=50)
                self.flann = cv.FlannBasedMatcher(indexParams, searchParams)

                # create a bowImgDescriptorExtractor
                self.extractBow = cv.BOWImgDescriptorExtractor(self.extract, self.flann)
                #create a bag-of-word K means trainer with 40 clusters
                self.bowKmeansTrainer = cv.BOWKMeansTrainer(40)


                try:
                    # insert the other typeid's first positive training image
                    # as the negative training samples for this typeid to bowKmeansTrainer
                    for j in range(1, STATES_NUM, 1):
                        if j != i:
                            self.bowKmeansTrainer.add(self.extract_sift(self._path(j, self.positive, 1)))

                    #insert positive training samples to bowKmeansTrainer   # which might be more than typeid
                    for j in range(self._trainingFrameCount):
                        self.bowKmeansTrainer.add(self.extract_sift(self._path(i, self.positive, j)))
                except Exception as e:
                    self.logger.error(e)
                    raise ValueError

                # generate vocabulary
                voc = self.bowKmeansTrainer.cluster()
                self.extractBow.setVocabulary(voc)

                bowFeature = self.extractBow.compute(graySnapshot, self.detector.detect(graySnapshot, self.interestedMask))

                pred = svm.predict(bowFeature)
                if pred[1][0][0] == 1.0:
                    cv.putText(currentImage,'matched with svm model {}'.format(i), (10,30),
                               cv.FONT_HERSHEY_COMPLEX,1,(0,255,0), 2, cv.LINE_AA)
                    cv.imshow('BOW + SVM success {}'.format(i), currentImage)

                    self.logger.info('currentImage matched with svm model {}'.format(i))
                else:
                    cv.putText(currentImage,'NOT matched with svm model {}'.format(i), (10,30),
                               cv.FONT_HERSHEY_COMPLEX,1,(0,0,255), 2, cv.LINE_AA)
                    cv.imshow('BOW + SVM failure {}'.format(i), currentImage)
                    self.logger.info('currentImage does NOT match with svm model {}'.format(i))
                cv.waitKey(10)

                i += 1

            cv.destroyAllWindows()





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
    parser.add_argument("--width", dest='width', help='set video camera width [1280,800,640,160 etc]',default=1920, type=int)
    parser.add_argument("--height", dest='height', help='set video camera height [960,600,480,120 etc]',default=1080, type=int)
    parser.add_argument("--duration", dest='duration',help='how many frames to discard before confirmation',default=30, type=int)
    parser.add_argument("--imgWidth", dest='imgWidth', help='set ROI image width [1280,800,640,etc]',default=600, type=int)
    parser.add_argument("--imgHeight", dest='imgHeight', help='set ROI image height [960,600,480,etc]',default=600, type=int)
    parser.add_argument("--folder", dest='folder',help='folder name to store training image files', default= "./trainingImages", type=str)
    parser.add_argument("--roiFolder", dest='roiFolder',help='folder name to store Region of Interest training image files', default ='./rois', type=str)
    parser.add_argument("--featureFolder", dest='featureFolder',help='folder name to store image feature files', default ='./features', type=str)
    parser.add_argument("--imageFormat", dest='imageFormat',help='image format [png,jpg,gif,jpeg]', default ='png', type=str)
    parser.add_argument("--skipCapture", dest='skipCapture',help='do not overwrite existing image files [0,1]', default = True, type=int)
    parser.add_argument("--modelFolder", dest='modelFolder',help='folder name to store trained SVM models', default = './models', type=str)

    args = parser.parse_args()


    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s %(message)s')
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logHandler)

    logger.info(args)


    solution = BuildDatabase(logger, "build database", args.deviceId,
                             range(1, STATES_NUM, 1), args.portId, args.duration,
                             args.width, args.height, args.imgWidth,args.imgHeight, args.folder, args.roiFolder,
                             args.featureFolder, imgFormat=args.imageFormat,
                             modelFolder=args.modelFolder, modelPrefixName='svmxml',skipCapture=args.skipCapture)
    try:
        solution.run()
    except ValueError:
        pass
    solution.reportTestResult()




