"""
this utility is main program used to test LCD display in manufacturing test.
it receive lcd check command via serial port from a controller,
 indicating id that is currently on display, then it takes a picture of the LCD display,
comparing it with preloaded predefined training image sets, if find a best match among
the training image set and the id matches with that training image filename, return  pass
 otherwise return fail
Project kickoff date: Nov 17,2021
coding start date: Nov 19,2021

phase 1: (completed on Dec 1)
design and coding completed to prove image capture and compare concept 
content: 1. threshTuneTrackbar.py to look for optimal parameters for edgeDetection,
        2. edgeDetect.py  to implement image capture/ROI isolate/warp image/ FLANN livecapture to one
            predefine target Image file compare
        3. lcdCheck.py : main OOP framework, argparser, capture sample images,logging,serialCom,

phase 2: development start on Dec 2,2021 
new feature, considering refactor code,
 1.support for multiple training images load/detectAndCompute feature,
 plan for future improvements to polish GUI in phase 3

 Dec 19:
 SVM models completes, predict accuracy to be improved.

phase 3:
 1. add a tkinter GUI ?
     
"""

