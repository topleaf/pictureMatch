import serial
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import subprocess
from filters import SharpenFilter
from edgeDetect import isolateROI

class WindowManager:
    def __init__(self,windowName, keyPressCallback):
        self._screenResolution = self._getCurrentScreenRes()
        self._windowName = windowName
        self.keyPressCallback = keyPressCallback
        self._isWindowCreated = False
        self._isDrawRect = False
        self._rectCords = None  # a rectangle range (x,y,w,h) inside this window
        self._keyPoints = None  # keypoints detected to be shown in this window
        pass

    def setKeypoints(self, keypoints):
        self._keyPoints = keypoints

    @property
    def rectCords(self):
        return self._rectCords

    def setRectCords(self, x, y, w, h):
        """
        set rectangle coordinations to be drawn in this window
        :param x:
        :param y:
        :param w:
        :param h:
        :return:
        """
        self._rectCords = x, y, w, h

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
        #draw a red rectangle over the region specified by self._rectCords
        if self._isDrawRect:
            if self.rectCords is not None:
                cv.rectangle(frame, (self._rectCords[0], self._rectCords[1]),
                          (self._rectCords[0] + self._rectCords[2]-1,
                             self._rectCords[1] + self._rectCords[3]-1), (0, 255, 0), 3)
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
                 snapWindowManager = None, shouldMirrorPreview=False, width=640, height=480,
                 compareResultList = [],  warpImgSize = (600,600)):
        self.logger = logger
        self._capture = None
        self._deviceId = deviceId
        self._cameraWidth = width
        self._cameraHeight = height
        self.previewWindowManager = previewWindowManger
        self.shouldMirrorPreview=shouldMirrorPreview
        self._snapWindowManager = snapWindowManager
        self._frame = None
        self._channel = 0
        self._imageFileName = None
        self._expectedModelId = None     # use current frame to  compare with which trained Svm model?
        self._svm = None
        self._bowExtractor = None
        self._interestedMask = None     # region of interest to be compared with trained model in self._frame
        self._enteredFrame = False
        self._compareResultList = compareResultList
        self._matchThreshold = 0.8
        self._warpImgSize = warpImgSize
        self._trainingImg = None       # expected training img, to be shown
        self.w = 0          # initial snapshot window left coordination

    def openCamera(self):
        self._capture = cv.VideoCapture(self._deviceId)
        if not self._capture.isOpened():
            raise Exception('Could not open video device {}'.format(self._deviceId))
        self._setCaptureResolution(self._cameraWidth, self._cameraHeight)

    @property
    def cameraIsOpened(self):
        return self._capture.isOpened()

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

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
            return self._frame

    @property
    def isWritingImage(self):
        return self._imageFileName is not None

    @property
    def isComparingTarget(self):
        return self._expectedModelId is not None


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

        # compare it with target trained model , if needed
        if self.isComparingTarget:
            self._compare()
            self._expectedModelId = None

        # self.logger.debug('in exitFrame(): get valid frame, display it ')
        # draw to the windowPreview , if any
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = np.fliplr(self._frame).copy()
                self.w, h = self.previewWindowManager.show(mirroredFrame, 10, 10, True)
            else:
                self.w, h = self.previewWindowManager.show(self._frame, 10, 10, True)

                # visualize the expectedModelId's first standard image in snapWindow
            if self._snapWindowManager is not None and self._trainingImg is not None:
                self._snapWindowManager.show(self._trainingImg.copy(), self.w, 10, True)



        # write to image file, if any
        if self.isWritingImage:
            self.logger.debug('in exitFrame(),write frame to file {}'.format(self._imageFileName))
            cv.imwrite(self._imageFileName, self._frame)
            self._snapWindowManager.show(self._frame, self.w, 10, True)
            self._imageFileName = None


            
        # release the frame
        self._frame = None
        self._enteredFrame = False

    def closeCamera(self):
        self._capture.release()

    def save(self, filename):
        """
        save the next frame to disk with the name filename
        :param filename:
        :return:  '0' success, '1' failed
        """
        self._imageFileName = filename
        return '0'

    def setCompareModel(self, expectedModelId, svmModelList,
                        bowExtractorsList, interestedMask,
                        trainingImg):
        """
        set SVM expectedModelId to be used to compare with current frame
        :param filename:
        :return:
        """
        self._expectedModelId = expectedModelId
        self._svm = svmModelList[self._expectedModelId-1]
        self._bowExtractor = bowExtractorsList[self._expectedModelId-1]
        self._interestedMask = interestedMask
        self._trainingImg = trainingImg


    # def _convert2bipolar(self, img):
    #     onedim = np.reshape(img, -1)
    #     m = int(onedim.mean())
    #     return cv.threshold(img, m, 255, cv.THRESH_BINARY)
    def _compare(self):
        """
        use designated svm model to predict current frame, give matched or not matched verdict
        show them on screen of snapshot window
        :return:
        """
        graySnapshot = cv.cvtColor(self._frame, cv.COLOR_BGR2GRAY)
        grayTrain = cv.cvtColor(self._trainingImg, cv.COLOR_BGR2GRAY)

        if graySnapshot is not None and grayTrain is not None\
            and self._svm is not None and self._bowExtractor is not None:
            #create  keypoints detector
            detector = cv.xfeatures2d.SIFT_create()

            #set keypoints in preview and snapshot window so they have this
            # information in case user pressed 'D' key to show them on window
            keypoints = detector.detect(graySnapshot, self._interestedMask)
            self.previewWindowManager.setKeypoints(keypoints)
            kp1 = detector.detect(grayTrain, self._interestedMask)
            self._snapWindowManager.setKeypoints(kp1)

            bowFeature = self._bowExtractor.compute(graySnapshot, keypoints)

            _, result = self._svm.predict(bowFeature)
            a, pred = self._svm.predict(bowFeature, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
            score = pred[0][0]
            self.logger.info('SVM model id: {}, Class: {:.1f}, Score:{:.4f}'.format(self._expectedModelId, result[0][0], score))
            if result[0][0] == 1.0:
                if score <= -0.98:
                    cv.putText(self._frame, 'match with svm model {},score is {:.4f}'
                               .format(self._expectedModelId, score), (10, graySnapshot.shape[0]-30),
                               cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), thickness=6)
                    self.logger.info('live image matched with svm model {}'.format(self._expectedModelId))
                else:
                    cv.putText(self._frame, 'NOT match with svm model {},score is {:.4f}'
                               .format(self._expectedModelId, score), (10, graySnapshot.shape[0]-30),
                               cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), thickness=6)

                    self.logger.info('currentImage does NOT match with svm model {}'.format(self._expectedModelId))
            # # visualize the live captured image in preview Window
            # w, h = self.previewWindowManager.show(self._frame.copy(), 10, 10, True)

        else:
            self.logger.warning('failed to retrieve a frame from camera,skip predicting')


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
    def __init__(self, logger, serialDeviceName, timeoutInSecond):
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
        # self._commandPrefix = b'\x52\x40'
        # self._commandBody = b'\x00\x00'

        self._commandPrefix = b'\x53\x50\x00\x0d'
        # self._commandBody = b'\x00\x00'
        self._expectedReponseLen = 18


    def _commandMapping(self, picId):
        """
        translate picId to control bytes sent to DUT
        :param picId:  int from 1-65535
        :return: 13-byte length bytes, lower 4 bits in every bytes is effective
        """
        controlInt = pow(2, picId) - 1
        controlBytes = []
        for i in range(13):
            lastFourBits = controlInt & 0x0F
            controlInt >>= 4
            controlBytes.insert(0, lastFourBits)

        command = self._commandPrefix
        for i in range(len(controlBytes)):
            command += controlBytes[i].to_bytes(1, 'big')
        return command

    def send(self, picId):
        """

        :param picId: int from 0-255
        :return: the bytes of command that is sent out
        """
        # newContent = (lambda x: 0x01 << x-1 if x > 0 else 0)(picId)  # 0: 00, 1: 01, 2:02, 3:04, 4:08

        # command = self._commandPrefix + picId.to_bytes(1, 'big')*13
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
