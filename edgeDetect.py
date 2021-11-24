import cv2 as cv
import numpy as np

img = cv.imread('./1.png')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('original',img)
cv.moveWindow('original', 10,100)

cv.imshow('gray',img_gray)
cv.moveWindow('gray', 400,100)

cv.imwrite('./pictures/1-canny.jpg', cv.Canny(img_gray,175,10))
cv.imshow('canny',cv.imread('./pictures/1-canny.jpg'))
cv.moveWindow('canny',800,100)

ret, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# ret, thresh = cv.threshold(img_gray,227,255,cv.THRESH_BINARY)
cv.imshow('thresh',thresh)
cv.moveWindow('thresh',1200,100)
print('ret = %d' %ret)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# print('contours={}'.format(contours))
# print('hierarchy={}'.format(hierarchy))
cv.drawContours(img, contours, -1, (0,255,0), 2)
cv.imshow('contours', img)
cv.moveWindow('contours', 10, 400)
# find inner rectangle
def findROI(hierarchy,contours):
    if hierarchy.shape[1] == 2:
        minx,miny = contours[1][0][0]
        maxx,maxy = contours[1][0][0]
        for cords in contours[1]:
            if cords[0,0] < minx:
                minx = cords[0,0]
            if cords[0,1] < miny:
                miny = cords[0,1]
            if cords[0,0] > maxx:
                maxx = cords[0,0]
            if cords[0,1] > maxy:
                maxy = cords[0,1]
        return minx,miny, maxx, maxy
    return 0,0,0,0

minx,miny, maxx,maxy = findROI(hierarchy,contours)
cv.rectangle(img,(minx,miny),(maxx, maxy),(255,0,0),2)
cv.imshow('detected ROI', img)



cv.waitKey()

cv.destroyAllWindows()

