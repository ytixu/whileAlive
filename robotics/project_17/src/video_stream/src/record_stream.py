#!/usr/bin/env python

import rospy
from stf_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

class record_streamer:

	def __init__(self, video_file):
		self.pub = rospy.Publisher('image', Image)
		self.bridge = CvBridge()
		self.video = cv2.VideoCapture(video_file)

	def publish(self):
		while(cap.isOpened()):

			# check cv_bridge tutorial


