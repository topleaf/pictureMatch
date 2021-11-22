"""
this utility is used to test LCD display in manufacturing test.
it sends command via serial port to a controller, indicating which lcd segment to be lighted up,
after receiving response from serial port, taking a picture of the LCD display,
comparing it with predefined display pattern, return a pass or fail verdict string after the
predefined test sequence completes
Project kickoff date: Nov 17,2021
coding start date: Nov 19,2021
"""

from managers import WindowManager,CaptureManager,CommunicationManager
import logging
import argparse
import time


class LcdTestSolution(object):
    def __init__(self,logger,windowName,captureDeviceId,predefinedPatterns,portId):
        """

        :param windowName: the title of window to show captured picture,str
        :param captureDeviceId:  the device id of the camera
        :param predefinedPatterns:
        :param portId: serial port id: int
        """

        self.logger = logger
        self._windowManager = WindowManager(windowName, self.onKeyPress)
        self._captureManager = CaptureManager(logger, captureDeviceId,self._windowManager,False)
        self._predefinedPatterns = predefinedPatterns
        self._expireSeconds = 5
        try:
            self._communicationManager = CommunicationManager(self.logger, '/dev/ttyUSB'+str(portId),
                                                              self._expireSeconds)
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




    def run(self,mode):
        for self._current in self._predefinedPatterns:
            response = b''
            retry = 0
            while response == b'' and retry < 3:
                self.logger.debug('send command to switch to {},retry={}'.format(self._current, retry))
                command = self._communicationManager.send(self._current)
                response = self._communicationManager.getResponse()
                if response == command:
                    self.logger.info('Press lcd switch key in 3 seconds to set a new image')
                    time.sleep(3)

                    self._compareComplete = False
                    while self._windowManager.isWindowCreated and not self._compareComplete:
                        self._captureManager.enterFrame()
                        if mode == 0:
                            # run in calibration mode, capture and save the next frame img to file
                            # in current directory by the name of self._current
                            self.__testResults.append(self._captureManager.
                                                      save(str(self._current)+'.png'))
                            # self._compareComplete = True
                        else:
                            # compare mode, load standard image file from disk and compare
                            # it with current image
                            self.__capturedPicture = self._captureManager.frame
                            if self.__capturedPicture is not None:
                                ret = self._captureManager.compare(self.__capturedPicture, self._current)
                                self.__testResults.append(ret)
                                # self._compareComplete = True
                            else:
                                self.logger.debug('in mode 1, retrieving the capture device returns empty frame')
                        self._captureManager.exitFrame()
                        self._windowManager.processEvents()


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


    def reportTestResult(self, mode):
        self._communicationManager.close()
        if mode == 0:
            self.logger.info('calibration completes:')
        else:
            self.logger.info('test completes:')
        self.logger.info('result is: {}'.format(self.__testResults))



    def onKeyPress(self,keyCode):
        if keyCode == ord('q') or keyCode == 27:
            self._windowManager.destroyWindow()
        elif keyCode == 32:  # space key
            self.logger.info('save screenshoot to snapshot.png')
            self._captureManager.save('snapshot.png')
        elif keyCode == ord('n') or keyCode == ord('N'):
            self._compareComplete = True
        else:
            self.logger.debug('unknown key {} pressed'.format(chr(keyCode)))

        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lcd manufacturing tester")
    parser.add_argument("--mode", dest='mode',help='calibration or test [0,1]',default=0, type=int)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s %(message)s')
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logHandler)

    logger.info(args)

    solution = LcdTestSolution(logger, "LCD test window", 0, [1, 2, 3], 0)
    solution.run(args.mode)
    solution.reportTestResult(args.mode)




