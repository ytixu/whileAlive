#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

MIN_MOTION = 333

class record_streamer:

	def __init__(self, subTopic):
		self.image_sub = rospy.Subscriber(subTopic,Image,self.callback)
		self.bridge = CvBridge()
		self.motion_pub = rospy.Publisher('motion_pub', Image, queue_size=10)
		# self.header = Header()
		self.lastImage = None

	def callback(self,data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)

		if self.lastImage is None:
			self.lastImage = gray
		else:
			firstFrame = self.lastImage
			frameDelta = cv2.absdiff(firstFrame, gray)
			thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

			# dilate the thresholded image to fill in holes, then find contours
			# on thresholded image
			thresh = cv2.dilate(thresh, None, iterations=2)
			(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)

			# loop over the contours
			for c in cnts:
				# if the contour is too small, ignore it
				if cv2.contourArea(c) < MIN_MOTION:
					continue

				# compute the bounding box for the contour, draw it on the frame,
				# and update the text
				(x, y, w, h) = cv2.boundingRect(c)
				cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

			self.lastImage = gray

			cv2.imshow("Motion", cv_image)

		cv2.waitKey(1)

	def destroy(self):
		cv2.destroyAllWindows()
