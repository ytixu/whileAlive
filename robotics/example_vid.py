import numpy as np
from matplotlib import pyplot as plt
import cv2

cap = cv2.VideoCapture('capture.avi')
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    i += 1

    # hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # if i % 100 == 0:
    imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    cv2.imshow('frame',frame)

    #     color = ('b','g','r')
    #     for i,col in enumerate(color):
    #         histr = cv2.calcHist([frame],[i],None,[256],[0,256])
    #         plt.plot(histr,color = col)
    #         plt.xlim([0,256])
    #     plt.show()
    # else:
    #     cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()