#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
# import imagesift

class record_streamer:

	def __init__(self, subTopic):
		self.image_sub = rospy.Subscriber(subTopic,Image,self.callback)
		self.bridge = CvBridge()
		self.hue_pub = rospy.Publisher('input/hue', Image, queue_size=10)
		self.sat_pub = rospy.Publisher('input/saturation', Image, queue_size=10)
		self.start_pub = rospy.Publisher('start', Header, queue_size=10)
		self.stop_pub = rospy.Publisher('stop', Header, queue_size=10)
		self.header = Header()
		self.started = False

	def callback(self,data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		# gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
		hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
		hue, sat, val = cv2.split(hsv_image);

		if self.started:
			self.header.stamp = rospy.Time.now()
			self.stop_pub.publish(self.header)
		else:
			self.started = True

		try:
			hue_image = self.bridge.cv2_to_imgmsg(hue, "mono8")
			sat_image = self.bridge.cv2_to_imgmsg(sat, "mono8")

			self.hue_pub.publish(hue_image)
			self.sat_pub.publish(sat_image)
		except CvBridgeError as e:
			print(e)

		self.header.stamp = rospy.Time.now()
		self.start_pub.publish(self.header)

		# print hue.shape

		# frames, desc = imagesift.get_sift_keypoints(gray_image)
		# cv_image = imagesift.draw_sift_frames(gray_image, frames)

		cv2.imshow("Hue window", hue)
		cv2.imshow("Sat window", sat)
		cv2.waitKey(1)

	def destroy(self):
		cv2.destroyAllWindows()
