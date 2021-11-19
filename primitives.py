import cv2 as cv
import numpy as np

def capture_photo(pic_name):
    cameraCapture = cv.VideoCapture(0)
    # fps=30 # assumption
    # pic_size=(int(cameraCapture.get(cv.CAP_PROP_FRAME_WIDTH)),
    #           int(cameraCapture.get(cv.CAP_PROP_FRAME_HEIGHT)))
    # cameraWriter = cv.VideoWriter(pic_name,cv.VideoWriter_fourcc('I','4','2','0'),
    #                               fps,pic_size)

    success,frame = cameraCapture.read()
    if success:
        cv.imwrite(pic_name,frame)
        print('saved to {}'.format(pic_name))
    cameraCapture.release()
    return frame



if __name__ =="__main__":
    print('this is to check lcd display')
    target_pic_filename = '/home/lijin/Pictures/led/500-473.png'
    test_pic_filename = 'home/lijin/Pictures/led/part1.png'

    img_target = cv.imread(target_pic_filename, 0)
    img_test = cv.imread(test_pic_filename, 0)
    frame = capture_photo('captured.png')
    cv.namedWindow('camera window')
    cv.imshow('camera window', frame)
    cv.waitKey()
    cv.destroyWindow('camera window')

    # cv.show(img_target)


