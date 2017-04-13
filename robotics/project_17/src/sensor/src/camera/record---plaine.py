#!/usr/bin/env python

import time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class record:

	def __init__(self):
		# self.image_sub = rospy.Subscriber(subTopic, Image, self.motion_callback)
		self.bridge = CvBridge()
		self.camera_pub = rospy.Publisher('camera_pub', Image, queue_size=10)

		self.cap = None

		rate = rospy.Rate(15) # 10hz
		while not rospy.is_shutdown() and self.spin():
			rate.sleep()

	def getNextFrame(self):
		if self.cap == None:
			self.cap = cv2.VideoCapture('/home/ytixu/gitHTML/whileAlive/robotics/ob5.avi')

		if type(self.cap) != type(1) and self.cap.isOpened():
			# self.cap.read()
			self.cap.read()
			return self.cap.read()
		else:
			self.cap.release()
			self.cap = 1

		return None, None

	def spin(self):
		ret, frame = self.getNextFrame()
		if frame == None:
			return False

		try:
			data = self.bridge.cv2_to_imgmsg(frame, "bgr8")
			self.camera_pub.publish(data)
		except CvBridgeError as e:
			print(e)

		return True

	def destroy(self):
		self.cap.release()

