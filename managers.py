import serial
import cv2 as cv

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
    def __init__(self,deviceId):

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
