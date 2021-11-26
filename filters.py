import cv2 as cv
import numpy as np

# cv.filter2D()
class VConvolutionFilter:
    def __init__(self, kernel):
        self.kernel = kernel
        pass

    def apply(self,src,dst):
        cv.filter2D(src,-1, kernel=self.kernel, dst=dst)

class SharpenFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[-1,-1,-1,-1,-1],
                           [-1,-1,-1,-1,-1],
                           [-1,-1, 25,-1,-1],
                           [-1,-1,-1,-1,-1],
                           [-1,-1,-1,-1,-1]]
                         )
        VConvolutionFilter.__init__(self, kernel)


class Embolden5Filter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[-4,-3,-2,-1, 0],
                           [-3,-2,-1, 0, 1],
                           [-2,-1, 1, 1, 2],
                           [-1, 0, 1, 2, 3],
                           [ 0, 1, 2, 3, 4]]
                          )
        VConvolutionFilter.__init__(self, kernel)

class Embolden3Filter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[-2,-1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]]
                          )
        VConvolutionFilter.__init__(self, kernel)
