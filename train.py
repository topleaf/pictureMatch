
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


    def run(self):
        #  step 1: capture and save all positive training images to respective image folders/pos-x
        self._captureAndSaveAllImages()
        #  step 2:  use some of other id's positive training images as one id's negative training images
        # self._createNegativeTrainingSamples()
        #  step 3: training each SVM models and save them to disk.
        self._trainSVMModels()

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
        else: # get the first positive sample of possible other typeid as this typeid's negative sample
            assert 0<=i<STATES_NUM
            negId = (typeid+i+1) % STATES_NUM
            fullpath="%s/%s%d.%s"  %(join(self._folderName, str(negId)), self.positive, 1, self._imgFormat)
            return fullpath


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

    def _createNegativeTrainingSamples(self):
        """
         get the first  items of other type positive images , use them as negative sample for this id
        :return:
        """
        pass

    def extract_sift(self,fn):
        """
        get descriptor of a image in a give path
        :param fn: full path of the image
        :return:
        """
        im  = cv.imread(fn, 0)
        return self.extract.compute(im, self.detector.detect(im))[1]


    def _bowFeatures(self,fn):
        """
        get descriptor of a given path image
        :param fn:  full path of the image
        :return:  its descriptor
        """
        im = cv.imread(fn,0)
        return self.extractBow.compute(im, self.detector.detect(im))


    def _trainSVMModels(self):

        #create  keypoints detector and descriptor extractor
        self.detector = cv.xfeatures2d.SIFT_create()
        self.extract = cv.xfeatures2d.SIFT_create()

        #create a flann matcher
        FLANN_INDEX_KDTREE = 0
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        searchParams = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(indexParams, searchParams)



        # training multiple svm models and save them to disk
        svmModels = []
        for self._current in self._predefinedPatterns:
            #create a bag-of-word K means trainer with 40 clusters
            self.bowKmeansTrainer = cv.BOWKMeansTrainer(40)

            # create a bowImgDescriptorExtractor
            self.extractBow = cv.BOWImgDescriptorExtractor(self.extract, self.flann)

            # insert negative training samples to bowKmeansTrainer
            for i in range(STATES_NUM):
                self.bowKmeansTrainer.add(self.extract_sift(self._path(self._current, self.negative, i)))

            #insert positive training samples to bowKmeansTrainer   # which might be more than typeid
            for i in range(self._trainingFrameCount):
                self.bowKmeansTrainer.add(self.extract_sift(self._path(self._current,self.positive,i)))

            # generate vocabulary
            voc = self.bowKmeansTrainer.cluster()
            self.extractBow.setVocabulary(voc)

            # create both positive and negative training samples lists
            trainData, trainLabels = [],[]
            for i in range(self._trainingFrameCount):
                trainData.extend(self._bowFeatures(self._path(self._current, self.positive, i)))
                trainLabels.append(1)

            for i in range(STATES_NUM):
                trainData.extend(self._bowFeatures(self._path(self._current,self.negative, i)))
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
            svmModels.append(svm)



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

STATES_NUM = 3

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
                             range(0, STATES_NUM, 1), args.portId, args.duration,
                             args.width, args.height, args.imgWidth,args.imgHeight, args.folder, args.roiFolder,
                             args.featureFolder, imgFormat=args.imageFormat,
                             modelFolder=args.modelFolder, modelPrefixName='svmxml',skipCapture=args.skipCapture)
    try:
        solution.run()
    except ValueError:
        pass
    solution.reportTestResult()




