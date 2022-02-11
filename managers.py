import serial
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import subprocess
from filters import SharpenFilter
from edgeDetect import warpImg, getRequiredContours
from skimage.metrics import structural_similarity as compare_ssim
SKIP_STATE_ID = 24      # skip id=24,  because its image is the same as 24
STATES_NUM = 52


class WindowManager:
    def __init__(self,windowName, keyPressCallback):
        self._screenResolution = self._getCurrentScreenRes()
        self._windowName = windowName
        self.keyPressCallback = keyPressCallback
        self._isWindowCreated = False
        self._isDrawRect = False
        self._rectCords = None  # list of  4 corners coordination [(SX, SY), (LB_X, LB_Y), (EX, EY), (RU_X, RU_Y)] inside this window
        self._keyPoints = None  # keypoints detected to be shown in this window
        pass

    def setKeypoints(self, keypoints):
        self._keyPoints = keypoints

    @property
    def rectCords(self):
        return self._rectCords

    def setRectCords(self, boxPoints):
        """
        set rectangle coordinations to be drawn in this window
        :param boxPoints : list of  4 corners coordination [(SX, SY), (LB_X, LB_Y), (EX, EY), (RU_X, RU_Y)]

        :return:
        """
        self._rectCords = np.array(boxPoints, np.int32)

    @property
    def isDrawRect(self):
        return self._isDrawRect

    def setDrawRect(self, value):
        if value in [True, False]:
            self._isDrawRect = value

    def _getCurrentScreenRes(self):
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


    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv.namedWindow(self._windowName, cv.WINDOW_NORMAL)
        self._isWindowCreated = True

    def show(self, frame, x, y, resize=False):
        """

        :param frame:
        :param x:
        :param y:
        :param resize: if resize to window to fit into display screen ratio, window takes 1/4 screen
        :return: width,height after resizing
        """
        cv.moveWindow(self._windowName, x, y)
        #draw a cyan rectangle over the region specified by self._rectCords, if in binary image,
        # it will display as light white color
        if self._isDrawRect:
            if self.rectCords is not None:
                cv.polylines(frame, [self._rectCords], True, (255, 255, 125), 3)
            if self._keyPoints is not None:
                frame = cv.drawKeypoints(image=frame, outImage=frame, keypoints=self._keyPoints, flags=4, color=(51,163,236))
        width = int(frame.shape[1])
        height = int(frame.shape[0])
        if resize:
            scaleX = self._screenResolution[0]/frame.shape[1]/2
            scaleY = self._screenResolution[1]/frame.shape[0]/2
            scale = min(scaleX, scaleY)
            width = int(frame.shape[1]*scale)
            height = int(frame.shape[0]*scale)
            cv.resizeWindow(self._windowName, width, height)
        cv.imshow(self._windowName, frame)
        return width, height

    def destroyWindow(self):
        cv.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv.waitKey(1)
        if self.keyPressCallback is not None and keycode != -1:
            #discard any non-ascii info encoded by GTK.
            keycode &= 0xFF
            self.keyPressCallback(keycode)

class CaptureManager:
    def __init__(self, logger, deviceId, previewWindowManger = None,
                 snapWindowManager = None, promptWindowManager=None, shouldMirrorPreview=False, width=640, height=480,
                 compareResultList = [],  warpImgSize = (600,600), blurLevel=9,
                 roiBox = [(0,0),(0,480),(640,0),(640,480)],cameraNoise=6,structureSimilarityThreshold=23,
                 offsetRangeX=5,offsetRangeY=5,deltaArea=40,deltaCenterX=20,deltaCenterY=20,deltaRadius=10):
        self.logger = logger
        self._capture = None
        self._deviceId = deviceId
        self._cameraWidth = width
        self._cameraHeight = height
        self.previewWindowManager = previewWindowManger
        self.shouldMirrorPreview=shouldMirrorPreview
        self._snapWindowManager = snapWindowManager
        self._promptWindowManager = promptWindowManager
        self._frame = None
        self._channel = 0
        self._imageFileName = None
        self._expectedTrainingImageId = None     # use current frame to  compare with which predefined training image id?
        self._svm = None
        self._bowExtractor = None
        self._interestedMask = None     # region of interest to be compared with trained model in self._frame
        self._enteredFrame = False
        self._compareResultList = compareResultList
        self._matchThreshold = 0.8
        self._warpImgSize = warpImgSize
        self._trainingImg = None       # expected training img, to be shown
        self.w = 0          # initial snapshot window left coordination
        self._showImageType = 0     # show original image without processing in live preview window
        # self._thresholdOffset = threshOffset  # user defined threshold offset value to change image to binary,EXPECIALLY IMPORTANT
        # self._thresholdValue = 0  # automatically calculated threshold value to change image to binary
        self._blurLevel = blurLevel     # user defined threshold value to change image to binary,EXPECIALLY IMPORTANT
        self._detector = cv.xfeatures2d.SIFT_create()  #SIFT detector
        self._box = roiBox
        self._previousFrame = None  #
        self._diffThresholdLevel = cameraNoise
        # if consecutive frames' gray level difference are less than noiseLevel, they are treated as the same
        self._enableSmooth = False  # disable noise suppressing during capturing and saving training sample phase
        self._ssim_diff_thresh_level = structureSimilarityThreshold
        self.offsetX = offsetRangeX   # range in x direction of allowing camera shift
        self.offsetY = offsetRangeY

        self._deltaArea = deltaArea     # maximum tolerance in contours area difference between live image and training sample image
        self._deltaCenterX = deltaCenterX  # maximum tolerance in contours center coordination difference
        self._deltaCenterY = deltaCenterY
        self._deltaRadius = deltaRadius    # maximum tolerance in contours' minEnclosingCircle radius difference
        self._userPrompt = None         # userPrompt in Str, to be displayed on screen if any


    def openCamera(self):
        self._capture = cv.VideoCapture(self._deviceId)
        if not self.cameraIsOpened:
            raise Exception('Could not open video device {}'.format(self._deviceId))
        self._setCaptureResolution(self._cameraWidth, self._cameraHeight)
        # set camera buffersize to 1
        self._capture.set(cv.CAP_PROP_BUFFERSIZE, 1)

    @property
    def displayImageType(self):
        return self._showImageType

    def setDisplayImageType(self, value):
        """
        set image display type to original, afterProcessed image , or warped region of interest image from original image
        :param value:
        :return:
        """
        if value in (0, 1, 2):
            self._showImageType = value
            self.logger.debug('setDisplayImageType to {}'.format(value))

    @property
    def cameraIsOpened(self):
        if self._capture is not None:
            return self._capture.isOpened()
        return False

    def _setCaptureResolution(self,width,height):
        """
        set video camera's resolution,
        :param width:  eg 1920
        :param height:  eg 1080
        :return:
        """
        res = self._capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
        res =self._capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)


    def _rescaleFrame(self,frame,percent=75):
        """

        :param frame:
        :param percent: target frame percentage to original
        :return:
        """
        width = int(frame.shape[1]*percent/100)
        height = int(frame.shape[0]*percent/100)
        dim = (width, height)

        return cv.resize(frame, dim, interpolation=cv.INTER_AREA)

    def _setFrameRes(self,frame, width=640, height=480):
        """

        :param frame:
        :param width:
        :param height:
        :return:
        """
        return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)


    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self,value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    def _smooth(self):
        """
        smooth, reduce noise according to diffThresholdLevel in self._box area

        :return:  smoothed frame
        """
        if self._frame is not None:
            prevGray = cv.cvtColor(self._previousFrame, cv.COLOR_BGR2GRAY)
            gray = cv.cvtColor(self._frame, cv.COLOR_BGR2GRAY)
            if self._blurLevel != 0:
                blurPrevFrame = cv.GaussianBlur(prevGray, (self._blurLevel, self._blurLevel), 0)
                blurFrame = cv.GaussianBlur(gray, (self._blurLevel,self._blurLevel), 0)
            else:
                blurPrevFrame = prevGray
                blurFrame = gray

            diff = cv.absdiff(blurFrame, blurPrevFrame)

            # project region of interest to new coordination for compare purpose only
            warpBlurPrev = warpImg(blurPrevFrame, self._box, self._warpImgSize[0], self._warpImgSize[1])
            warpBlurFrame = warpImg(blurFrame, self._box, self._warpImgSize[0], self._warpImgSize[1])
            diffRoi = cv.absdiff(warpBlurPrev, warpBlurFrame)
            # self._previousFrame = self._frame   # set current Frame to be previousFrame for next time comparasion
            if diffRoi.max() <= self._diffThresholdLevel:      # variation in roi is less than noise threshold leovel
                # return self._previousFrame      # use previousFrame, discard current frame as a noise frame
                # combine current frame and previousFrame in gray format, copy ROI content of previousFrame into
                # this smoothed frame
                newGray = np.where(diff > self._diffThresholdLevel, gray, prevGray)
                newColor = cv.cvtColor(newGray, cv.COLOR_GRAY2BGR)
                return newColor
            else:
                self._previousFrame = self._frame   # set current Frame to be previousFrame for next time comparasion
                return self._frame
        else:
            return self._frame

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            if self._previousFrame is None:
                _, self._previousFrame = self._capture.retrieve()
                self._frame = self._previousFrame
            else:       # smooth frame according to diffThresholdLevel
                _, self._frame = self._capture.retrieve()
                if self._enableSmooth:      # need to suppress noise, in predicting phase
                    self._frame = self._smooth()
            return self._frame


    @property
    def isWritingImage(self):
        return self._imageFileName is not None

    @property
    def isComparingTarget(self):
        return self._expectedTrainingImageId is not None

    @property
    def hasUserPrompt(self):
        return self._userPrompt is not None

    def setUserPrompt(self, testResultList, prompt):
        """

        :param testResultList:  list of [0,1,... ...]
        :param prompt:  str , to be displayed on screen, or None,
        :return:
        """
        if prompt is None:  # clear content
            self._userPrompt = prompt
            return

        failureItems = []
        for i in range(len(testResultList)):
            if testResultList[i] == 0:
                failureItems.append(i+1)

        self._userPrompt = dict(verdict='', color=(0, 255, 0))
        if len(failureItems) != 0:
            self._userPrompt['verdict'] = 'FAIL! ({})'.format(failureItems)
            self._userPrompt['color'] = (0, 0, 255)
        else:
            self._userPrompt['verdict'] = "PASS,"
            self._userPrompt['color'] = (0, 255, 0)

        self._userPrompt['verdict'] += prompt

    def enterFrame(self):
        """
        capture the next frame, if any
        :return:
        """
        assert not self._enteredFrame, 'previous enterFrame() has no matching exitFrame()'
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()
            # self.logger.debug('in enterFrame:grab() returns {}'.format(self._enteredFrame))

    def exitFrame(self):
        """
        draw to window, write to files , release the frame
        :return: None , if not successfully retrieve a frame
        """
        # check whether any grabbed frame is retrievable.
        # the getter may retrieve and cache the frame
        if self.frame is None:
            self._enteredFrame = False
            self.logger.warning('in exitFrame: self._frame is None, retrieve an empty frame from camera')
            return None

        if self.hasUserPrompt:
            cv.putText(self._frame, self._userPrompt['verdict'], (10, 100),
                       cv.FONT_HERSHEY_COMPLEX, 2, self._userPrompt['color'], thickness=2)

        compareResult = None
        # compare it with specified training image , if needed
        if self.isComparingTarget:
            compareResult = self._compare()
            self._expectedTrainingImageId = None

        # self.logger.debug('in exitFrame(): get valid frame, display it ')
        # draw to the windowPreview , if any
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = np.fliplr(self._frame).copy()
                self.w, h = self.previewWindowManager.show(mirroredFrame, 10, 10, True)
            else:
                self.w, h = self.previewWindowManager.show(self._frame.copy(), 10, 10, True)

                # visualize the expectedModelId's first standard image in snapWindow
            if self._snapWindowManager is not None and self._trainingImg is not None:
                self._snapWindowManager.show(self._trainingImg.copy(), self.w+50, 10, True)



        # write to image file, if any
        if self.isWritingImage:
            self.logger.debug('in exitFrame(),write frame to file {}'.format(self._imageFileName))
            cv.imwrite(self._imageFileName, self._frame)
            self._snapWindowManager.show(self._frame, self.w, 10, True)
            self._imageFileName = None


            
        # release the frame
        self._frame = None
        self._enteredFrame = False
        return compareResult

    def closeCamera(self):
        self._capture.release()

    def save(self, filename):
        """
        save the next frame to disk with the name filename
        :param filename:
        :return:  None
        """
        self._imageFileName = filename

    def setCompareModel(self, expectedTrainingImageId, interestedMask,
                        trainingImg,enableSmooth):
        """

        :param expectedTrainingImageId:
        :param interestedMask:
        :param trainingImg:
        :param enableSmooth:
        :return:
        """

        self._expectedTrainingImageId = expectedTrainingImageId
        self._interestedMask = interestedMask
        self._trainingImg = trainingImg
        self._enableSmooth = enableSmooth


    def preProcess(self, image):
        """
        binarize original image according to defined threshold value and blurLevel
        :param image: original image in bgr mode
        :return: thresh and blurred image with defined preprocess parameters
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # first trial, only do threshold, result is OK
        # ret, binary = cv.threshold(gray,self._thresholdValue, 255, cv.THRESH_BINARY_INV)
        # return cv.blur(binary, (self._blurLevel, self._blurLevel))

        # second trial, add erode and dilate to further smoothing image,removing camera pixel dance
        # result is good with STATES_NUM 52, self.BOW_CLUSTER_NUM=STATES_NUM*10, duration = 20 (sample training pic) per class
        # training time period = 1497 seconds
        # ret, thresh_img = cv.threshold(gray, self._thresholdValue, 255, cv.THRESH_BINARY_INV)
        # blur = cv.blur(thresh_img, (self._blurLevel, self._blurLevel))
        # imgErode = cv.erode(blur, kernel=np.ones((3, 3)), iterations=1)
        # imgDilate = cv.dilate(imgErode, kernel=np.ones((3, 3)), iterations=1)

        #third trial, at first use smooth in CaptureManager to remove camera noise, then
        # use gaussian blur,threshold,dilate and erode
        # blur = cv.GaussianBlur(gray, (self._blurLevel, self._blurLevel), 0)
        # ret, thresh_img = cv.threshold(blur, self._thresholdValue, 255, cv.THRESH_BINARY_INV)
        # imgDilate = cv.dilate(thresh_img, kernel=np.ones((3, 3)), iterations=1)
        # imgErode = cv.erode(imgDilate, kernel=np.ones((3, 3)), iterations=1)
        # return imgErode

        #fourth trial, at first use smooth in CaptureManager to remove camera noise, then
        # use gaussian blur only, use structure similarity to compare
        blur = cv.GaussianBlur(gray, (self._blurLevel, self._blurLevel), 0)

        return blur


    # def _compare(self):
    #     """
    #     use designated svm model to predict current frame, give matched or not matched verdict
    #     show them on screen of snapshot window
    #     :return:
    #     """
    #     # graySnapshot = cv.cvtColor(self._frame, cv.COLOR_BGR2GRAY)
    #     # ret, threshSnapshot = cv.threshold(graySnapshot, 84, 255, cv.THRESH_BINARY_INV)
    #     compareResult = dict(matched=False, predictedClassId=None, score=None)
    #
    #     blurSnapshot = self.preProcess(self._frame)
    #     blurTrain = self.preProcess(self._trainingImg)
    #     # warp the interested ROI
    #     warpBlurSnapshot = warpImg(blurSnapshot, self._box, self._warpImgSize[0], self._warpImgSize[1])
    #     warpBlurTrain = warpImg(blurTrain, self._box, self._warpImgSize[0], self._warpImgSize[1])
    #
    #     if warpBlurSnapshot is not None and warpBlurTrain is not None\
    #         and self._svm is not None and self._bowExtractor is not None:
    #         #create  keypoints detector
    #
    #         #set keypoints in preview and snapshot window so they have this
    #         # information in case user pressed 'D' key to show them on window
    #         keypoints = self._detector.detect(warpBlurSnapshot, self._interestedMask)
    #         self.previewWindowManager.setKeypoints(keypoints)
    #
    #         kp1 = self._detector.detect(warpBlurTrain, self._interestedMask)
    #         self._snapWindowManager.setKeypoints(kp1)
    #
    #         bowFeature = self._bowExtractor.compute(warpBlurSnapshot, keypoints)
    #         if bowFeature is None:
    #             self.logger.warning('empty SIFT extracted from current image')
    #             return compareResult
    #
    #         _, result = self._svm.predict(bowFeature)
    #         a, pred = self._svm.predict(bowFeature, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
    #         score = pred[0][0]
    #         self.logger.info('SVM model id:{}, Class:{:.1f}, Score:{:.4f}'.format(self._expectedTrainingImageId, result[0][0], score))
    #
    #         if self._showImageType == 1:
    #             self._frame = blurSnapshot
    #             self._trainingImg = blurTrain
    #         elif self._showImageType == 2:
    #             self._frame = warpImg(self._frame, self._box, self._warpImgSize[0], self._warpImgSize[1])
    #             self._trainingImg = warpImg(self._trainingImg, self._box, self._warpImgSize[0], self._warpImgSize[1])
    #         if self._expectedTrainingImageId != 0: ## using single classification model
    #             if result[0][0] == 1.0:
    #                 if score <= -0.99:
    #                     cv.putText(self._frame, 'match model {},score is {:.4f}'
    #                                .format(self._expectedTrainingImageId, score), (10, self._frame.shape[0]-30),
    #                                cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), thickness=6)
    #                     self.logger.info('live image matched with svm model {}'.format(self._expectedTrainingImageId))
    #                     compareResult['matched'] = True
    #
    #                 else:
    #                     cv.putText(self._frame, 'might NOT match svm model {},score is {:.4f}'
    #                                .format(self._expectedTrainingImageId, score), (10, self._frame.shape[0]-30),
    #                                cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), thickness=6)
    #                     self.logger.info('currentImage does NOT match with svm model {}'.format(self._expectedTrainingImageId))
    #                     compareResult['matched'] = False
    #                 compareResult['predictedClassId'] = self._expectedTrainingImageId
    #                 compareResult['score'] = score
    #             elif result[0][0] == -1.0:  # not matched
    #                 compareResult['matched'] = False
    #                 compareResult['predictedClassId'] = self._expectedTrainingImageId
    #                 compareResult['score'] = score
    #                 if score >= 0.99:
    #                     cv.putText(self._frame, 'Sure Not match with svm model {},score is {:.4f}'
    #                                .format(self._expectedTrainingImageId, score), (10, self._frame.shape[0]-30),
    #                                cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), thickness=6)
    #                     self.logger.info('live image NOT match with svm model {}'.format(self._expectedTrainingImageId))
    #                 else:
    #                     cv.putText(self._frame, 'NOT match with svm model {},score is {:.4f}'
    #                                .format(self._expectedTrainingImageId, score), (10, self._frame.shape[0]-30),
    #                                cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), thickness=6)
    #
    #                     self.logger.info('currentImage does NOT match with svm model {}'.format(self._expectedTrainingImageId))
    #         else:       # use unique multiclassification model to predict
    #             predictedClassId = result[0][0]
    #             if predictedClassId in range(1, STATES_NUM+1, 1):
    #                 cv.putText(self._frame, 'match class {}'
    #                            .format(int(predictedClassId)), (10, self._frame.shape[0]-30),
    #                            cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), thickness=6)
    #                 self.logger.debug('live image matched with class {},score ={:.4f}'.
    #                                  format(int(predictedClassId), score))
    #                 # if predictedClassId in [1, 2, 51, 52]:
    #                 #     self.logger.info('bowFeature{}={}'.format(predictedClassId, bowFeature))
    #                 compareResult['matched'] = True
    #             else:  # not matched
    #                 compareResult['matched'] = False
    #                 if score >= 0.99:
    #                     cv.putText(self._frame, 'Sure No match with any training images,score={:.4f}'
    #                                .format(score), (10, self._frame.shape[0]-30),
    #                                cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), thickness=6)
    #                     self.logger.info('live image NOT match!!!')
    #                 else:
    #                     cv.putText(self._frame, 'NOT match with svm model,score is {:.4f}'
    #                                .format(score), (10, self._frame.shape[0]-30),
    #                                cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), thickness=6)
    #                     self.logger.info('currentImage does NOT match with svm model')
    #
    #             compareResult['predictedClassId'] = int(predictedClassId)
    #             compareResult['score'] = score
    #     else:
    #         self.logger.warning('failed to retrieve a frame from camera,skip predicting')
    #
    #     return compareResult

    def _tryShiftCamera(self, blurSnapshot,blurTrain, xOffset,yOffset):
        """
        using slide-window of step size 1 pixel to try in the range of [xOffset,yOffset] area
        trying to search if there is a position offset where blurSnapshot matches with blurTrain
        in the area of ROI
        :param blurSnapshot: live image after preprocess
        :param blurTrain: training image after preprocess
        :param xOffset:  pixel number to shift in x direction
        :param yOffset:  pixel numbers to shift in y direction
        :return: contourLen: minimum contour length in all trials
                threshImage: structure similarity compare returns difference image
                score: structure similairty compare score , range (0,1)
        """
        contourLen = 99
        thresh = None
        score = 0.0
        # warp the interested ROI
        warpBlurTrain = warpImg(blurTrain, self._box, self._warpImgSize[0], self._warpImgSize[1])
        for j in range(yOffset):
            for i in range(xOffset):
                # setup  the simulated interested box after camera is shifted in x and y direction
                boxCameraShift = self._box + [i, j]
                warpBlurSnapshot = warpImg(blurSnapshot, boxCameraShift, self._warpImgSize[0], self._warpImgSize[1])

                if warpBlurSnapshot is not None and warpBlurTrain is not None:
                    # use self._interestedMask to fetch only interested area data
                    warpBlurSnapshot = np.where(self._interestedMask == 0, 0, warpBlurSnapshot)
                    warpBlurTrain = np.where(self._interestedMask == 0, 0, warpBlurTrain)

                    # compare structure similarity
                    score, diff = compare_ssim(warpBlurSnapshot, warpBlurTrain, full=True)
                    diff = (diff*255).astype('uint8')

                    #find contours of difference between 2  warpImages, if len(cnts) > 0, then 2 warpImages are different
                    thresh = cv.threshold(diff, 255-self._ssim_diff_thresh_level, 255, cv.THRESH_BINARY_INV)[1]  #cv2.THRESH_OTSU
                    cnts, hierachy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    currentLen=len(cnts)
                    self.logger.debug('i={},j={},warp imgs structure similarity currentLen ={}'.format(i,j,currentLen))
                    if currentLen == 0:  # found a match, exit loop
                        #set keypoints in preview and snapshot window so they have this
                        # information in case user pressed 'D' key to show them on window
                        keypoints = self._detector.detect(warpBlurSnapshot, self._interestedMask)
                        self.previewWindowManager.setKeypoints(keypoints)

                        kp1 = self._detector.detect(warpBlurTrain, self._interestedMask)
                        self._snapWindowManager.setKeypoints(kp1)
                        contourLen = 0
                        self.logger.info('offset_x={},offset_y={}'.format(i, j))
                        return contourLen, thresh, score
                    elif currentLen < contourLen:
                        contourLen = currentLen
                else:
                    break

        return contourLen, thresh, score


    # def _compare(self):
    #     """
    #     compare designated training image with current active frame using structure similarity method,
    #      give matched or not matched verdict
    #     show them on screen of snapshot window
    #     :return:
    #     """
    #     compareResult = dict(matched=False, predictedClassId=None, score=None)
    #
    #     blurSnapshot = self.preProcess(self._frame)
    #     blurTrain = self.preProcess(self._trainingImg)
    #     cnts_len, thresh, score = self._tryShiftCamera(blurSnapshot, blurTrain, self.offsetX, self.offsetY)
    #     if self._showImageType == 1:
    #         self._frame = blurSnapshot
    #         self._trainingImg = blurTrain
    #     elif self._showImageType == 2:
    #         self._frame = thresh if thresh is not None else warpImg(self._frame, self._box, self._warpImgSize[0], self._warpImgSize[1])
    #         self._trainingImg = warpImg(self._trainingImg, self._box, self._warpImgSize[0], self._warpImgSize[1])
    #     if self._expectedTrainingImageId != 0:
    #         if cnts_len == 0:   # do we find any contours in thresh image ?
    #             cv.putText(self._frame, 'match sample {},score is {:.4f},SS_threshold={}'
    #                        .format(self._expectedTrainingImageId, score,self._ssim_diff_thresh_level), (10, self._frame.shape[0]-30),
    #                        cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), thickness=6)
    #             self.logger.info('live image matched with training sample {}'.format(self._expectedTrainingImageId))
    #             compareResult['matched'] = True
    #         else:
    #             cv.putText(self._frame, 'NOT match sample {},score is {:.4f},SS_threshold={}'
    #                        .format(self._expectedTrainingImageId, score,self._ssim_diff_thresh_level), (10, self._frame.shape[0]-30),
    #                        cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), thickness=6)
    #             self.logger.info('live image NOT matched with training sample {},cnts len={}'.
    #                              format(self._expectedTrainingImageId, cnts_len))
    #             compareResult['matched'] = False
    #         compareResult['predictedClassId'] = self._expectedTrainingImageId
    #         compareResult['score'] = score
    #     else:
    #         self.logger.info('impossible branch')
    #         raise ValueError
    #
    #     return compareResult

    # def _calcThreshValue(self, img, interestMask,blurr_level, offset ):
    #     """
    #     automatically selecting a threshold value to binarize the image
    #     :param img: original img with  BGR 3 channels
    #     :param interestMask:
    #     :param blurr_level: int
    #     :param offset:  int
    #     :return:  new thresholdValue
    #
    #     """
    #     # auto calculating appropriate threshold value for this frame
    #     # use interestedMask to fetch only interested area data, set pixels outside interestedMask to 255
    #     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     blur = cv.GaussianBlur(gray, (blurr_level, blurr_level), sigmaX=1, sigmaY=1)
    #     imgROI = np.where(interestMask == 0, 255, blur)
    #     thresholdValue = np.min(imgROI) + offset
    #
    #     return thresholdValue

    def _compare(self):
        """
        compare designated training image with current active frame using contours similarity
        if both images' contour numbers, their areas, their minEnclosingCircle center and radius are within
        a predefined range , then give matched verdict . this method is more flexible than structure similarity
        if both images are shifted within the predefined range, they are treated as equal.
        show them on screen of snapshot window
        :return:
        """
        compareResult = dict(matched=True, predictedClassId=None)

        # get the lcd Screen part rectangle from original img
        # warp the interested ROI

        snapShotImg = self._frame.copy()
        trainImg = self._trainingImg.copy()

        imgWarp = warpImg(snapShotImg, self._box,  self._warpImgSize[0], self._warpImgSize[1])
        # auto calculating appropriate threshold value for this frame
        # self._thresholdValue = self._calcThreshValue(imgWarp, self._interestedMask, self._blurLevel, self._thresholdOffset)

        imgThresh, conts, imgWarp = getRequiredContours(imgWarp, self._blurLevel, 25, 125,
                                                   1, 1,
                                                   np.ones((13, 13)), self._interestedMask,
                                                   minArea=100, maxArea=50000,
                                                   cornerNumber=4, draw=self._showImageType,
                                                   returnErodeImage=False)


        warpTrain = warpImg(trainImg, self._box, self._warpImgSize[0], self._warpImgSize[1])
        # auto calculating appropriate threshold value for this training frame
        # self._thresholdValue = self._calcThreshValue(warpTrain, self._interestedMask, self._blurLevel, self._thresholdOffset)
        imgTrainThresh, contsTrain, warpTrain = getRequiredContours(warpTrain, self._blurLevel, 25, 125,
                                                   1, 1,
                                                   np.ones((13, 13)), self._interestedMask,
                                                   minArea=100, maxArea=50000,
                                                   cornerNumber=4, draw=self._showImageType,
                                                   returnErodeImage=False)
        if self._showImageType == 1:
            # cv.rectangle(snapShotImg, self._box[0], self._box[2], (0,255,0), 2)
            self._frame = imgWarp
            self._trainingImg = warpTrain
        elif self._showImageType == 2:
            self._frame = imgThresh
            self._trainingImg = imgTrainThresh

        if self._expectedTrainingImageId != 0:
            lenConts = len(conts)
            lenContsTrain = len(contsTrain)
            if lenConts != lenContsTrain:       # do we have same contours  number?
                compareResult['matched'] = False
                self.logger.info('not match, lenConts={},lenContsTrain={}'.format(lenConts,lenContsTrain))
                cv.putText(self._frame, 'NOT match sample {},live={},train={}'
                           .format(self._expectedTrainingImageId,len(conts),len(contsTrain)), (10, self._frame.shape[0]-30),
                           cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), thickness=6)
            else:
                diffArea,diffCenterX,diffCenterY,diffRadius = 0,0,0,0
                for i in range(lenConts):
                    #  check both images' all contours,
                    #  each contours' area, minEnclosingCircle's center and radius must be within range
                    # before we give a matched verdict
                    area, center, radius = conts[i][0],conts[i][1],conts[i][2]
                    areaT, centerT, radiusT = contsTrain[i][0],contsTrain[i][1], contsTrain[i][2]
                    diffArea = abs(area-areaT)
                    diffCenterX = abs(center[0]-centerT[0])
                    diffCenterY = abs(center[1]-centerT[1])
                    diffRadius = abs(radius-radiusT)
                    self.logger.debug('diffArea={},diffCenter=({},{}),radius={}'.format(diffArea, diffCenterX,
                                                                                   diffCenterY, diffRadius))
                    if diffArea > self._deltaArea or diffRadius > self._deltaRadius \
                            or diffCenterX > self._deltaCenterX or diffCenterY > self._deltaCenterY:
                        compareResult['matched'] = False
                        break
                if compareResult['matched']:
                    cv.putText(self._frame, 'match sample {},diff={},({},{}),{}'
                               .format(self._expectedTrainingImageId, diffArea,diffCenterX,diffCenterY,diffRadius), (10, self._frame.shape[0]-30),
                               cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), thickness=6)
                    self.logger.info('live image matched with training sample {}'.format(self._expectedTrainingImageId))
                else:
                    cv.putText(self._frame, 'NOT match sample {},diff={},({},{}),{}'
                               .format(self._expectedTrainingImageId,diffArea,diffCenterX,diffCenterY,diffRadius), (10, self._frame.shape[0]-30),
                               cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), thickness=6)
                    self.logger.info('live image NOT matched with training sample {}'.
                                     format(self._expectedTrainingImageId))
            compareResult['predictedClassId'] = self._expectedTrainingImageId
        else:
            self.logger.info('impossible branch')
            raise ValueError

        return compareResult




    # def _compare(self):
    #     """
    #     use cv.matchTemplate to compare targetImg with current frame
    #     append maxloc,minloc,maxval,minval to self._compareResultList
    #     threshold is likelyhood of matched criteria, between 0.0 - 1.0,
    #     the larger, the stricter
    #     :return:
    #     """
    #     targetImg = cv.imread(self._targetFileName)
    #
    #     # get image of the ROI in target picture, removing unrelated background,
    #     # projecting the ROI to (wP,hP) size coordination system
    #     targetImg = isolateROI(targetImg, False, False,
    #                            wP=self._warpImgSize[0], hP=self._warpImgSize[1])
    #     cv.imwrite(self._targetFileName.split('.')[0]+'_targetROI.png', targetImg)
    #
    #     templateImg = isolateROI(self._frame, False,True,
    #                              wP=self._warpImgSize[0], hP=self._warpImgSize[1])
    #     cv.imwrite(self._targetFileName.split('.')[0]+'_template.png', templateImg)
    #
    #     result = cv.matchTemplate(targetImg, templateImg, cv.TM_CCOEFF_NORMED)
    #
    #     minval, maxval, minloc, maxloc = cv.minMaxLoc(result)
    #
    #     yloc, xloc = np.where(result >= self._matchThreshold)
    #     self.logger.info('maxval = {},length of xloc is {}'.format(maxval, len(xloc)))
    #     rectangles = []
    #     for (x, y) in zip(xloc, yloc):
    #         rectangles.append((int(x), int(y), self._warpImgSize[0],self._warpImgSize[1]))
    #         rectangles.append((int(x), int(y), self._warpImgSize[0],self._warpImgSize[1]))
    #     rectangles, weights = cv.groupRectangles(rectangles, 1, 0.2)
    #     for (x, y, w, h) in rectangles:
    #         cv.rectangle(targetImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #
    #     # cv.imwrite(self._targetFileName.split('.')[0]+"_result.png", result)
    #     cv.imwrite(self._targetFileName.split('.')[0]+"_match.png", targetImg)
    #     self._compareResultList.append(maxval)
    #

    # def _compare(self):
    #     """
    #     compare current frame with predefined targetFileName and return metric of likelyhood
    #     between 0 and 1, the larger , both pictures are more likely to be the same
    #     :return:
    #     """
    #     self.logger.debug('in _compare(), compare frame with file {}'.format(self._targetFileName))
    #     targetImg = cv.imread(self._targetFileName)
    #
    #     targetImgGray = cv.cvtColor(targetImg, cv.COLOR_BGR2GRAY)
    #     frameGray = cv.cvtColor(self._frame, cv.COLOR_BGR2GRAY)
    #
    #     # step 1: thresh both images using mean threshold
    #     # frameMean, frameThresh = self._convert2bipolar(frameGray)
    #     # targetMean, targetImgThresh = self._convert2bipolar(targetImgGray)
    #
    #     _, frameThresh = cv.threshold(frameGray, 30, 255, cv.THRESH_BINARY_INV )
    #     _, targetImgThresh = cv.threshold(targetImgGray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    #     # step 2: find contours of both images
    #     contoursFrame, hierarchyFrame = cv.findContours(frameThresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #     contoursTarget, hierarchyTarget = cv.findContours(targetImgThresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #
    #     cv.drawContours(self._frame, contoursFrame, -1, (0,255,0), 2)
    #
    #     # step 3: get cords of ROI, which is the inner rectangle of the contours , to be fixed later
    #     minx,miny, maxx,maxy = self._findROI(hierarchyFrame,contoursFrame)
    #     cv.rectangle(self._frame,(minx,miny),(maxx, maxy),(255,0,0),2)
    #
    #     self._snapWindowManager.show(self._frame, self._frame.shape[1]+10, 10)
    #
    #
    #     # step 3: compare ROI of both image
    #     #
    #     targetImgBinary = np.array(targetImgGray)
    #     # after using matplotlib.pyplot.imsave or cv.imwrite to write file, the file becomes BGR or RGB format if reading
    #     # back them using matplotlib.pyplot.imread or cv.imread  3 channels
    #     # plt.imsave('targetImghsv.png', targetImgGray,cmap='gray',vmin=0,vmax=255)
    #     # cv.imwrite('targetImghsv_cv.png',targetImgGray)
    #     #plt.imsave('framehsv.png',frameGray,cmap='gray',vmin=0,vmax=255)
    #     frameFileName = 'frame.png'
    #     frameGrayedFileName = 'frame_grayed_cv.png'
    #     self.logger.debug('save both captured frame and grayed frame to files {},{}'.format(frameFileName,frameGrayedFileName))
    #     cv.imwrite(frameFileName, self._frame)
    #     cv.imwrite(frameGrayedFileName, frameGray)
    #
    #     # display it in snapshot window
    #     # self.previewWindowManager.show(targetImg,120,80)
    #     # self._snapWindowManager.show(frameGray, frameGray.shape[1]+10, 10)
    #
    #     #calculate the histogram and normalize it
    #     targetHist = cv.calcHist([targetImgGray], [0], None, [256], [0, 256])
    #     cv.normalize(targetHist,targetHist,alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    #     frameHist = cv.calcHist([frameGray], [0], None, [256], [0, 256])
    #     cv.normalize(frameHist, frameHist,alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    #
    #
    #
    #     #find and return the metric value , between 0 and 1, the larger the value, the more likely
    #     metric = cv.compareHist(targetHist, frameHist, cv.HISTCMP_BHATTACHARYYA)
    #     self.logger.info('metric value = %.3f' % metric)
    #     return metric

class CommunicationManager:
    """
    class to handle serial communication with controller SW, baud 38600, N, 8,1

    command format: 0x53 0x50 0x00 0x0d pic_id*13
    response format: 0x53 0x50 0x00 0x0d pic_id*13 checksum

    where checksum is the sum of previous 5 bytes ,pic_id is [0x00,0xFF]
    """
    def __init__(self, logger, serialDeviceName, timeoutInSecond, algorithm):
        self.logger = logger
        # initial serial port with 2 timeout, ser.send() and ser.read() will return after sending/receiving
        # or timeout
        try:
            self._ser = serial.Serial(serialDeviceName, 115200, timeout=timeoutInSecond,
                                      write_timeout=timeoutInSecond)
        except serial.SerialException as e:
            self.logger.error('failed to open serial device {}: {}'.format(serialDeviceName, e))
            raise ConnectionError
            return
        self._timer = timeoutInSecond
        self._commandPrefix = b'\x53\x50\x00\x0d'
        # self._commandBody = b'\x00\x00'
        self._expectedReponseLen = 18
        self._commands=[]
        self._algorithm=algorithm
        if self._algorithm == 3:
            self.buildAlgorithm3CommandList()


    def _commandMapping(self, picId):
        """
        translate picId to control bytes sent to DUT
        :param picId:  int from 1-65535
        :param algorithm: int to differentiate which method to use
        :return: 13-byte length bytes, lower 4 bits in every bytes is effective
        """
        command = self._commandPrefix
        if self._algorithm in [0, 1]:  # 0 :previous set bits are kept , incrementally 1: only one bit is set
            controlInt = pow(2, picId) - (1-self._algorithm)
            if self._algorithm == 1:
                controlInt >>= 1
            controlBytes = []
            for i in range(13):
                lastFourBits = controlInt & 0x0F
                controlInt >>= 4
                controlBytes.insert(0, lastFourBits)
            for i in range(len(controlBytes)):
                command += controlBytes[i].to_bytes(1, 'big')
        elif self._algorithm==2:
            divider, moder = divmod(picId, 4)
            # newContent = (lambda x: 0x01 << x-1 if x > 0 else 0)(moder)  # 0: 00, 1: 01, 2:02, 3:04, 4:08
            moder = (lambda x: 4 if x == 0 else x)(moder)
            controlInt = pow(2, moder)-1
            for i in range(13):
                if i == divider:
                    command += controlInt.to_bytes(1, 'big')
                else:
                    command += b'\x00'
        else:
            command = self._commands[picId-1]
        return command

    def buildAlgorithm3CommandList(self):

        for i in range(13):
            for j in range(i+1, 13):
                if i != j:
                    command = self._commandPrefix
                    xcommand = [b'\x00',b'\x00',b'\x00',b'\x00',b'\x00',b'\x00',b'\x00',b'\x00',
                                b'\x00',b'\x00',b'\x00',b'\x00',b'\x00']
                    xcommand[i] = b'\x0f'
                    xcommand[j] = b'\x0f'
                    for c in xcommand:
                        command += c
                    self._commands.append(command)


    def send(self, picId):
        """

        :param picId: int from 0-255
        :return: the bytes of command that is sent out
        """
        # newContent = (lambda x: 0x01 << x-1 if x > 0 else 0)(picId)  # 0: 00, 1: 01, 2:02, 3:04, 4:08

        # command = self._commandPrefix + newContent.to_bytes(1, 'big')*13
        command = self._commandMapping(picId)
        retLen = self._ser.write(command)
        self._expectedReponseLen = retLen+1
        return command





    def getResponse(self):
        """
         read  serial port and fetch input from serial port, it will
         block until get required length of bytes or  after timeout expires(return with whatever it received)
        :return: bytes
        """
        return self._ser.read(self._expectedReponseLen)

    def close(self):
        self._ser.close()

    def resetInputBuffer(self):
        self._ser.reset_input_buffer()

    def resetOutputBuffer(self):
        self._ser.reset_output_buffer()
