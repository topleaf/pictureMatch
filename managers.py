import serial
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class WindowManager:
    def __init__(self,windowName, keyPressCallback):
        self._windowName = windowName
        self.keyPressCallback = keyPressCallback
        self._isWindowCreated = False
        pass

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv.namedWindow(self._windowName, cv.WINDOW_NORMAL)
        self._isWindowCreated = True

    def show(self, frame, x, y):
        cv.moveWindow(self._windowName, x, y)
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
                 snapWindowManager = None, shouldMirrorPreview=False):
        self.logger = logger
        self._capture = cv.VideoCapture(deviceId)
        self.previewWindowManager = previewWindowManger
        self.shouldMirrorPreview=shouldMirrorPreview
        self._snapWindowManager = snapWindowManager
        self._frame = None
        self._channel = 0
        self._imageFileName = None
        self._targetFileName = None     # use current frame to  compare with which image file?
        self._enteredFrame = False

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
            self._snapWindowManager.show(self._frame, self._frame.shape[1]+10, 10)
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

    def setCompareFile(self,filename):
        """
        set targetFileName to be loaded and compare with current frame
        :param filename:
        :return:
        """
        self._targetFileName = str(filename) + '.png'

    def _compare(self):
        """
        compare current frame with predefined targetFileName and return metric of likelyhood
        between 0 and 1, the larger , both pictures are more likely to be the same
        :return:
        """
        self.logger.debug('in _compare(), compare frame with file {}'.format(self._targetFileName))
        targetImg = cv.imread(self._targetFileName)
        targetImgHsv = cv.cvtColor(targetImg, cv.COLOR_BGR2GRAY)
        frameHsv = cv.cvtColor(self._frame, cv.COLOR_BGR2GRAY)

        # after using matplotlib.pyplot.imsave or cv.imwrite to write file, the file becomes BGR or RGB format if reading
        # back them using matplotlib.pyplot.imread or cv.imread  3 channels
        # plt.imsave('targetImghsv.png', targetImgHsv,cmap='gray',vmin=0,vmax=255)
        # cv.imwrite('targetImghsv_cv.png',targetImgHsv)
        #plt.imsave('framehsv.png',frameHsv,cmap='gray',vmin=0,vmax=255)
        frameFileName = 'frame.png'
        frameGrayedFileName = 'frame_grayed_cv.png'
        self.logger.debug('save both captured frame and grayed frame to files {},{}'.format(frameFileName,frameGrayedFileName))
        cv.imwrite(frameFileName, self._frame)
        cv.imwrite(frameGrayedFileName, frameHsv)

        # display it in snapshot window
        # self.previewWindowManager.show(targetImg,120,80)
        self._snapWindowManager.show(frameHsv, frameHsv.shape[1]+10, 10)

        #calculate the histogram and normalize it
        targetHist = cv.calcHist([targetImgHsv], [0], None, [256], [0, 256])
        cv.normalize(targetHist,targetHist,alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        frameHist = cv.calcHist([frameHsv], [0], None, [256], [0, 256])
        cv.normalize(frameHist, frameHist,alpha=0, beta=255, norm_type=cv.NORM_MINMAX)



        #find and return the metric value , between 0 and 1, the larger the value, the more likely
        metric = cv.compareHist(targetHist, frameHist, cv.HISTCMP_BHATTACHARYYA)
        self.logger.info('metric value = %.3f' % metric)
        return metric

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
