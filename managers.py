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
        pass

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
        :param resize: if resize to window to fit into display screen ratio
        :return:
        """
        cv.moveWindow(self._windowName, x, y)
        if resize:
            scaleX = self._screenResolution[0]/frame.shape[1]
            scaleY = self._screenResolution[0]/frame.shape[0]
            scale = min(scaleX, scaleY)
            width = int(frame.shape[1]*scale)
            height = int(frame.shape[0]*scale)
            cv.resizeWindow(self._windowName, width, height)
        cv.imshow(self._windowName, frame)

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
        self._capture = cv.VideoCapture(deviceId)
        self._setCaptureResolution(width, height)
        self.previewWindowManager = previewWindowManger
        self.shouldMirrorPreview=shouldMirrorPreview
        self._snapWindowManager = snapWindowManager
        self._frame = None
        self._channel = 0
        self._imageFileName = None
        self._targetFileName = None     # use current frame to  compare with which image file?
        self._enteredFrame = False
        self._compareResultList = compareResultList
        self._matchThreshold = 0.8
        self._warpImgSize = warpImgSize

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
        return self._targetFileName is not None

    def enterFrame(self):
        """
        capture the next frame, if any
        :return:
        """
        assert not self._enteredFrame, 'previous enterFrame() has no matching exitFrame()'
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()
            self.logger.debug('in enterFrame:grab() returns {}'.format(self._enteredFrame))

    def exitFrame(self):
        """
        draw to window, write to files , release the frame
        :return:
        """
        # check whether any grabbed frame is retrievable.
        # the getter may retrieve and cache the frame
        if self.frame is None:
            self._enteredFrame = False
            self.logger.warning('in exitFrame: self._frame is None, retrieve an empty frame from camera')
            return

        # self.logger.debug('in exitFrame(): get valid frame, display it ')
        # draw to the windowPreview , if any
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = np.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame, 10, 10)
            else:
                self.previewWindowManager.show(self._frame, 10, 10)

        # write to image file, if any
        if self.isWritingImage:
            self.logger.debug('in exitFrame(),write frame to file {}'.format(self._imageFileName))
            cv.imwrite(self._imageFileName, self._frame)
            self._snapWindowManager.show(self._frame, self._frame.shape[1]+410, 10)
            self._imageFileName = None


        # compare it with target image , if needed
        if self.isComparingTarget:
            self._compare()
            self._targetFileName = None
            
        # release the frame
        self._frame = None
        self._enteredFrame = False

    def save(self, filename):
        """
        save the next frame to disk with the name filename
        :param filename:
        :return:  '0' success, '1' failed
        """
        self._imageFileName = filename
        return '0'

    def setCompareFile(self, filename):
        """
        set targetFileName to be loaded and compare with current frame
        :param filename:
        :return:
        """
        self._targetFileName = str(filename) + '.png'

    def _convert2bipolar(self, img):
        onedim = np.reshape(img, -1)
        m = int(onedim.mean())
        return cv.threshold(img, m, 255, cv.THRESH_BINARY)


    def _compare(self):
        """
        use cv.matchTemplate to compare targetImg with current frame
        append maxloc,minloc,maxval,minval to self._compareResultList
        threshold is likelyhood of matched criteria, between 0.0 - 1.0,
        the larger, the stricter
        :return:
        """
        targetImg = cv.imread(self._targetFileName)

        # get image of the ROI in target picture, removing unrelated background,
        # projecting the ROI to (wP,hP) size coordination system
        targetImg = isolateROI(targetImg, False, False,
                               wP=self._warpImgSize[0], hP=self._warpImgSize[1])
        cv.imwrite(self._targetFileName.split('.')[0]+'_targetROI.png', targetImg)

        templateImg = isolateROI(self._frame, False,True,
                                 wP=self._warpImgSize[0], hP=self._warpImgSize[1])
        cv.imwrite(self._targetFileName.split('.')[0]+'_template.png', templateImg)

        result = cv.matchTemplate(targetImg, templateImg, cv.TM_CCOEFF_NORMED)

        minval, maxval, minloc, maxloc = cv.minMaxLoc(result)

        yloc, xloc = np.where(result >= self._matchThreshold)
        self.logger.info('maxval = {},length of xloc is {}'.format(maxval, len(xloc)))
        rectangles = []
        for (x, y) in zip(xloc, yloc):
            rectangles.append((int(x), int(y), self._warpImgSize[0],self._warpImgSize[1]))
            rectangles.append((int(x), int(y), self._warpImgSize[0],self._warpImgSize[1]))
        rectangles, weights = cv.groupRectangles(rectangles, 1, 0.2)
        for (x, y, w, h) in rectangles:
            cv.rectangle(targetImg, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # cv.imwrite(self._targetFileName.split('.')[0]+"_result.png", result)
        cv.imwrite(self._targetFileName.split('.')[0]+"_match.png", targetImg)
        self._compareResultList.append(maxval)


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

    command format: 0x52 0x40 pic_id 0x00 0x00 checksum
    response format: 0x52 0x40 pic_id 0x00 0x00 checksum

    where checksum is the sum of previous 5 bytes ,pic_id is [0x00,0xFF]
    """
    def __init__(self, logger, serialDeviceName, timeoutInSecond):
        self.logger = logger
        # initial serial port with 2 timeout, ser.send() and ser.read() will return after sending/receiving
        # or timeout
        try:
            self._ser = serial.Serial(serialDeviceName, 38400, timeout=timeoutInSecond,
                                      write_timeout=timeoutInSecond)
        except serial.SerialException as e:
            self.logger.error('failed to open serial device {}: {}'.format(serialDeviceName, e))
            raise ConnectionError
            return
        self._timer = timeoutInSecond
        self._commandPrefix = b'\x52\x40'
        self._commandBody = b'\x00\x00'

    def _composeCommandWithCheckSum(self,content):
        """
        calculate checksum of the content, append it to the content bytes as checksum

        :param self:
        :param content: bytes
        :return:  bytes with checksum included
        """
        checksum = 0
        for i in range(len(content)):
            checksum += content[i]

        checksum &= 0xFF
        checksum = checksum.to_bytes(1, 'big')
        content += checksum
        return content

    def send(self, picId):
        """

        :param picId: int from 0-255
        :return: the bytes of command that is sent out
        """
        command = self._commandPrefix + picId.to_bytes(1, 'big') + self._commandBody
        command = self._composeCommandWithCheckSum(command)
        retLen = self._ser.write(command)
        return command

    def getResponse(self):
        """
         read  serial port and fetch input from serial port, it will
         block until get required length of bytes or  after timeout expires(return with whatever it received)
        :return: bytes
        """
        return self._ser.read(6)

    def close(self):
        self._ser.close()
