import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from sensor.msg import SensorImages
from cv_bridge import CvBridge, CvBridgeError

BACKGROUND_acc_weight = 0.05
BACKGROUND_history = 10
BACKGROUND_nGauss = 5
BACKGROUND_bgThresh = 0.5
BACKGROUND_noise = 10

class camera:

    def __init__(self, subTopic):
        self.image_sub = rospy.Subscriber(subTopic, Image, self.motion_callback)
        self.bridge = CvBridge()

        self.prvs = None
        self.fgbg = cv2.BackgroundSubtractorMOG(BACKGROUND_history,
                    BACKGROUND_nGauss,BACKGROUND_bgThresh,BACKGROUND_noise)

    def background_subtract(self, image):
        fgmask = self.fgbg.apply(image, learningRate=BACKGROUND_acc_weight)

        cv2.imshow('bg',fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            return

    def optical_flow(self, image):
        if self.prvs == None:
            frame1 = image
            self.prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
            self.hsv = np.zeros_like(frame1)
            self.hsv[...,1] = 255

        frame2 = image
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(self.prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        self.hsv[...,0] = ang*180/np.pi/2
        self.hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(self.hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow('opticalfb',rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            return
        elif k == ord('s'):
            cv2.imshow('opticalhsv', frame2)
            # cv2.imshow('opticalhsv', rgb)
            cv2.waitKey(1)
            # cv2.imwrite('opticalfb.png',frame2)
            # cv2.imwrite('opticalhsv.png',rgb)
        self.prvs = next


    def motion_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.background_subtract(cv_image)
        self.optical_flow(cv_image)

    def destroy(self):
        cv2.destroyAllWindows()