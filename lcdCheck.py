"""
this utility is used to test LCD display in manufacturing test.
it sends command via serial port to a controller, indicating which lcd segment to be lighted up,
after receiving response from serial port, taking a picture of the LCD display,
comparing it with predefined display pattern, return a pass or fail verdict string after the
predefined test sequence completes
Project kickoff date: Nov 17,2021
coding start date: Nov 19,2021
"""

import cv2 as cv
import numpy as np
from enum import Enum
from managers import WindowManager,CaptureManager,CommunicationManager
import logging
import time
from threading import Thread,currentThread

class Message(Enum):
    ACKNOWLEDGE = 0
    NACK = 1


class LcdTestSolution(object):
    def __init__(self,logger,windowName:str,captureDeviceId:int,predefinedPatterns:list,portId:int):
        """

        :param windowName: the title of window to show captured picture,str
        :param captureDeviceId:  the device id of the camera
        :param predefinedPatterns:
        :param portId: serial port id: int
        """

        self.logger = logger
        self._windowManager = WindowManager(windowName, self.onKeyPress)
        self._captureManager = CaptureManager(captureDeviceId)
        self._predefinedPatterns = predefinedPatterns
        self._expireSeconds = 5
        try:
            self._communicationManager = CommunicationManager(self.logger, '/dev/ttyUSB0', self._expireSeconds)
        except ConnectionError:
            self.logger.error('Abort!! Please make sure serial port is ready then retry')
            exit(-1)
        self.__capturedPicture = None
        self._windowManager.createWindow()
        self.__testResults = []
        self._current = None
        self._compareComplete = False
        self._comTimeout = False            # communication timeout ?
        self._comThread = None      # the thread of serial communication




    def run(self):
        # # start communication thread
        # self._comThread = Thread(target = communicationFunc, name="SerialCom Thread", daemon=True,
        #                          args=(self._communicationManager, self.logger))
        # self._comThread.start()

        for self._current in self._predefinedPatterns:
            response = b''
            retry = 0
            while response == b'' and retry < 3:
                self.logger.debug('send command to switch to {},retry={}'.format(self._current, retry))
                command = self._communicationManager.send(self._current)
                response = self._communicationManager.getResponse()
                if response == command:
                    self._compareComplete = False
                    while not self._compareComplete:
                        self._captureManager.enterFrame()
                        self.__capturedPicture = self._captureManager.frame
                        self.__testResults.append(self.__compareManager.compare(
                            self.__capturedPicture, self._current))
                        self._captureManager.exitFrame()
                        self._compareComplete = True
                elif response == b'':
                    self.logger.debug('no response after few seconds timeout')
                    retry += 1
                else:
                    self.logger.error("Receive a mismatched response from controller,skip this pic :{}".format(response))
                    self.__testResults.append('O')
                    self._compareComplete = True
                    break

            if retry == 3:
                self.logger.debug('pic_id:{},communication timeout,no response from controller after 3 times retry!'.format(self._current))
                self.__testResults.append('N')
                self._compareComplete = True
                continue


    def reportTestResult(self):
        self._communicationManager.close()
        self.logger.info('test result is: {}'.format(self.__testResults))



    def onKeyPress(self):
        pass


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s %(message)s')
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logHandler)
    solution = LcdTestSolution(logger, "LCD test window",0,[1,2,3],1)
    solution.run()
    solution.reportTestResult()




