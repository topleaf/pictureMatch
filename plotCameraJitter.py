import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import argparse

class CameraJitterPlot:
    def __init__(self, deviceId=0, frameNum=5,save = False,
                 sampleInteval=1, resW=800, resH=600,folderName='plotFrames',x=100,y=200,
                 liveMode = True):
        self.liveMode = liveMode


        self.frameNum = frameNum
        self.sampleInterval = sampleInteval
        self.folderName = os.getcwd()+'/'+folderName.lstrip('/').rstrip('/')+'/'
        self.windowName = 'live image window'
        self.gaps = []
        self.pixelValues = []
        self.x = x
        self.y = y
        self.save = save
        self.actualCount = 0
        if self.liveMode:
            self.camera = cv.VideoCapture(deviceId)
            self.camera.set(3,resW)
            self.camera.set(4,resH)
        else:
            self.frame = cv.imread(self.folderName+'0.png')
        pass

    def capture(self):
        count = 0
        try:
            os.mkdir(self.folderName)
        except:
            pass
        previousTimestamp  = time.time()
        while count < self.frameNum:
            if self.liveMode:
                succ, liveFrame = self.camera.read()
            else:
                succ = True
                liveFrame = self.frame
            currentTimestamp = time.time()
            if succ:
                gap = (currentTimestamp-previousTimestamp)
                self.gaps.append(gap)
                fps = 1/gap
                self.pixelValues.append(np.array(liveFrame)[self.y, self.x])
                if self.save:
                    cv.putText(liveFrame,'count={},FPS={:.0f}'.format(count,fps),(10,20),cv.FONT_HERSHEY_COMPLEX,
                               1, (0, 255, 0), 2)
                    cv.imwrite(self.folderName+str(count) + '.png', liveFrame)

                cv.imshow(self.windowName, liveFrame)
            previousTimestamp = currentTimestamp
            count += 1
            key = cv.waitKey(self.sampleInterval)
            key = key & 0xFF
            if key == 27:
                # cv.destroyWindow(self.windowName)
                break
        self.actualCount = count
        print('captured {} frames, sampleInterval is {} ms, saved to {}'.format(count,self.sampleInterval, self.folderName))

        if self.liveMode:
            self.camera.release()

    def show(self):
        """
        plot R,G,B values over count or over gap
        :param x:
        :param y:
        :return:
        """
        x = np.array(range(self.actualCount))
        x1 = np.array(self.gaps)
        y = np.array(self.pixelValues)
        b = [b[0] for b in y]
        g = [g[1] for g in y]
        r  = [r[2] for r in y]
        b = np.array(b)
        g = np.array(g)
        r = np.array(r)

        plt.title('live Capture Mode = {}, pixel position: ({},{})'.format((lambda x:'True' if x else 'False')(self.liveMode), self.x, self.y))
        plt.scatter(x, b, color='blue')
        plt.scatter(x, g, color='green')
        plt.scatter(x, r, color='red')
        # plt.scatter(x, x1, color='cyan')
        plt.xlabel('sample count #')
        plt.ylabel('r/g/b value or gap value')

        print('gaps value range is [{},{}] seconds, mean is {},std={}'.format(np.min(x1), np.max(x1),np.mean(x1),np.std(x1)))
        print('blue value range is [{},{}],mean is {},std={}'.format(np.min(b), np.max(b),np.mean(b),np.std(b)))
        print('green value range is [{},{}],mean is {},std={}'.format(np.min(g), np.max(g),np.mean(g),np.std(g)))
        print('red value range is [{},{}],mean is {},std={}'.format(np.min(r), np.max(r),np.mean(r),np.std(r)))
        plt.show()
        cv.destroyAllWindows()

        pass


if __name__ =='__main__':
    parser = argparse.ArgumentParser("plot the (b,g,r) values at position of (x,y) from live camera")
    parser.add_argument('--deviceId',dest="deviceId",help="video camera id: ",type=int,default=0)
    parser.add_argument('--x',dest="x",help="x position ",type=int,default=332)
    parser.add_argument('--y',dest="y",help="y position ",type=int,default=364)
    parser.add_argument('--frameNum',dest="frameNum",help="capture how many frames ",type=int,default=100)
    parser.add_argument('--save',dest="save",help="save to disk or not ",type=int,default=0)
    parser.add_argument('--liveMode',dest="live",help="live or playback mode ",type=int,default=1)
    args = parser.parse_args()
    plotter = CameraJitterPlot(deviceId=args.deviceId,save=args.save, frameNum=args.frameNum,x=args.x, y=args.y,
                               liveMode=args.live)#x=332, y=364
    plotter.capture()
    plotter.show()