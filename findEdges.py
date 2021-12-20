"""
algorithm to find a block's mass center and border with different color from
background color
"""
import cv2 as cv
import numpy as np

from scipy.ndimage import measurements

def find_sudoku_edges(im,axis, threshold_gray_level,ratio):
    """

    :param im:
    :param axis:
    :param threshold_gray_level: the threshold of gray_level
    :param ratio: interested region's sum of gray-level in x,y direction
            takes how much percentage of the whole picture in that direction,
            before it's accepted
    :return:
    """
    # turn gray image into binary image according to threshold_gray_level
    trim = 1*(im < threshold_gray_level)
    cv.imshow('trim', np.array(trim, dtype=np.uint8))
    s = trim.sum(axis=axis)
    # if axis == 0:
    #     cv.imshow('s-axis0', np.array(s.T, np.uint8))
    # else:
    #     cv.imshow('s-axis1', np.array(s, np.uint8))

    s = s > (ratio * max(s))
    # s = (s == max(s))
    cv.imshow('s>{:.2f} maxs'.format(ratio), np.array(s, np.uint8))
    #looking for connected domains
    s_labels, s_nbr = measurements.label(s)

    # cv.imshow('s_labels', np.array(s_labels,np.uint8))

    #calculate center of mass for all connected domains
    m = measurements.center_of_mass(s, labels=s_labels, index=range(1, s_nbr+1))
    m_int = [int(item[0]) for item in m]
    return m_int

if __name__ =='__main__':
    origin = cv.imread(f'/home/lijin/Pictures/摄像头/2021-11-27-132943.jpg')
    # origin = np.ones((10, 20, 3), dtype=np.uint8)
    # origin[:,:] = (255,0,255)
    # for i in range(0, origin.shape[0], origin.shape[0]//5):
    #     for j in range(0, origin.shape[1]):
    #         origin[i,j] = (0,0,0)



    while (True):
        cv.imshow('origin', origin)
        im_gray = cv.cvtColor(origin, cv.COLOR_BGR2GRAY)
        cv.imshow('gray', im_gray)
        # need to carefully select different threshold_gray_level,according
        # to different origin image.
        # investigate your origin image beforehand,
        x = find_sudoku_edges(im_gray, axis=0,threshold_gray_level=175,ratio=0.8)
        y = find_sudoku_edges(im_gray, axis=1,threshold_gray_level=175,ratio=0.8)

        draw_pic = origin.copy()
        for i in range(len(x)):
            for j in range(len(y)):
                origin[y[j],x[i]] = (255,255,255)  # draw mass center location in white
                if i == 0:
                    left = 0
                    w = x[i]
                else:
                    left = x[i-1] + w
                if j==0:
                    top = 0
                    h = y[j]
                    w_new = x[i] - left
                else:
                    top = y[j-1]+h
                    h = y[j] - top
                cv.rectangle(draw_pic,(left,top),(left+2*w_new-1,top+2*h-1),(0,0,255),2)
                if j == len(y)-1:
                    w = w_new

        cv.imshow('draw_pic',draw_pic)

        print('x={} , y = {}'.format(x,y))
        key = cv.waitKey(10)
        if key == 27:
            cv.destroyAllWindows()
            break
