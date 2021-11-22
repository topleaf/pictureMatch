import serial
import cv2 as cv
import numpy as np

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
        cv.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self,frame):
        cv.imshow(self._windowName,frame)

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
                 shouldMirrorPreview=False):
        self.logger = logger
        self._capture = cv.VideoCapture(deviceId)
        self.previewWindowManager = previewWindowManger
        self.shouldMirrorPreview=shouldMirrorPreview
        self._frame = None
        self._channel = 0
        self._imageFileName = None
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
            self.logger.debug('in exitFrame: self._frame is None, do not process empty frame')
            self._frame = None
            return

        # draw to the windowPreview , if any
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = np.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)

        # write to image file, if any
        if self.isWritingImage:
            cv.imwrite(self._imageFileName, self._frame)
            self._imageFileName = None

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
        # try:
        #     cv.imwrite(str(filename),captured_frame)
        # except Exception as e:
        #     self.logger.error('failed to save frame to %s: %s ' %(filename,e))
        #     return '1'
        # return '0'

    def compare(self,captured_frame,filename):
        self.logger.debug('in compare')
        standardImg = cv.imread(str(filename)+'.png')

        #convert to hsv
        standardHsv = cv.cvtColor(standardImg, cv.COLOR_BGR2HSV)
        testHsv = cv.cvtColor(captured_frame, cv.COLOR_BGR2HSV)

        #calculate histograms and normalized them
        histStandard = cv.calcHist([standardHsv],[0,1],None,[180,256],[0,180,0,256])
        cv.normalize(histStandard,histStandard,alpha=0,beta=1,norm_type=cv.NORM_MINMAX)

        histTest = cv.calcHist([testHsv],[0,1],None,[180,256],[0,180,0,256])
        cv.normalize(histTest,histTest,alpha=0,beta=1,norm_type=cv.NORM_MINMAX)

        #find the metric
        metric = cv.compareHist(histStandard,histTest,cv.HISTCMP_BHATTACHARYYA)
        self.logger.info('diff_histogram is %.3f' % metric)


        # cv.matchTemplate(testGray,standardGray,cv.)
        return '1'
        pass


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
