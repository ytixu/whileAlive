#!/usr/bin/env python


# Motion detection
# http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
#

import rospy
import cv2
from sensor_msgs.msg import Image
# from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

MIN_MOTION = 333
OBS_BOX = 33
SPIN_RATE = 0.33

class camera:

	def __init__(self, subTopic):
		self.image_sub = rospy.Subscriber(subTopic,Image,self.motion_callback)
		self.bridge = CvBridge()
		self.motion_pub = rospy.Publisher('motion_pub', Image, queue_size=10)
		self.snapshot_pub = rospy.Publisher('snapshot', Image, queue_size=10)
		# self.h = Header()
		self.lastImage = None
		self.acc_motion = None

	def motion_callback(self,data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		self.snapshot = cv_image

		gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)

		if self.lastImage is None:
			self.lastImage = gray
		else:
			firstFrame = self.lastImage
			# compute the absolute difference between the current frame and
			# first frame
			frameDelta = cv2.absdiff(firstFrame, gray)
			thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

			# dilate the thresholded image to fill in holes, then find contours
			# on thresholded image
			thresh = cv2.dilate(thresh, None, iterations=2)
			(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)

			has_motion = False
			# loop over the contours
			for c in cnts:
				# if the contour is too small, ignore it
				if cv2.contourArea(c) < MIN_MOTION:
					continue
				has_motion = True
				# # compute the bounding box for the contour, draw it on the frame,
				# # and update the text
				# (x, y, w, h) = cv2.boundingRect(c)
				# cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

			self.lastImage = gray

			cv2.imshow("Motion", thresh)
			cv2.waitKey(1)

			if has_motion:
				try:
					pub_image = self.bridge.cv2_to_imgmsg(thresh, "mono8")
				except CvBridgeError as e:
					print(e)
				self.motion_pub.publish(pub_image)
				self.snapshot_pub.publish(data)

	def destroy(self):
		cv2.destroyAllWindows()
