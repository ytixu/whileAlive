#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class record_viewer:

	def __init__(self):
		self.image_sub = rospy.Subscriber("image_raw",Image,self.callback)
		self.bridge = CvBridge()

	def callback(self,data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		# (rows,cols,channels) = cv_image.shape
		# if cols > 60 and rows > 60 :
		# 	cv2.circle(cv_image, (50,50), 10, 255)

		cv2.imshow("Image window", cv_image)

	def destroy(self):
		cv2.destroyAllWindows()

