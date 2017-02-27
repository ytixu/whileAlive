#!/usr/bin/env python


# Motion detection
#
# Image diff
# http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
#
# Background sub
# http://docs.opencv.org/trunk/db/d5c/tutorial_py_bg_subtraction.html

import rospy
import cv2
from sensor_msgs.msg import Image
# from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

MIN_MOTION = 333
OBS_BOX = 33
SPIN_RATE = 0.33

class camera:

	def __init__(self, subTopic, method=1):
		self.image_sub = rospy.Subscriber(subTopic,Image,self.motion_callback)
		self.bridge = CvBridge()
		self.motion_pub = rospy.Publisher('motion_pub', Image, queue_size=10)
		self.snapshot_pub = rospy.Publisher('snapshot', Image, queue_size=10)
		# self.h = Header()

		if method == 1:
			# background subtraction
			self.avg1 = None
			self.avg2 = None
		else:
			self.lastImage = None
		self.method = method

	def image_diff(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
			for i, c in enumerate(cnts):
				# if the contour is too small, ignore it
				if cv2.contourArea(c) < MIN_MOTION:
					continue
				has_motion = True
				# # compute the bounding box for the contour, draw it on the frame,
				# # and update the text
				# (x, y, w, h) = cv2.boundingRect(c)
				# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
			# cv2.drawContours(image, cnts, -1, (0, 255, 0), thickness=-1)

			self.lastImage = gray

			if has_motion:
				return thresh

		return None

	def background_substract(self, image):
		if self.avg1 == None:
			self.avg1 = np.float32(image)
			self.avg2 = np.float32(image)

		else:
			cv2.accumulateWeighted(image,avg1,0.1)
		    cv2.accumulateWeighted(image,avg2,0.01)

		    res1 = cv2.convertScaleAbs(avg1)
		    res2 = cv2.convertScaleAbs(avg2)

		    # cv2.imshow('img',f)
		    # cv2.imshow('avg1',res1)
		    # cv2.imshow('avg2',res2)
		    return rest2
		return None


	def motion_callback(self, data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		motion_image = None

		if self.method == 1:
			self.background_substract(cv_image)
		else:
			motion_image = self.image_diff(cv_image)

		if motion_image != None:
			cv2.imshow("Motion", motion_image)
			# cv2.imshow("Contour", cv_image)
			cv2.waitKey(1)

			try:
				pub_image = self.bridge.cv2_to_imgmsg(motion_image, "mono8")
			except CvBridgeError as e:
				print(e)

			self.motion_pub.publish(pub_image)
			self.snapshot_pub.publish(data)

	def destroy(self):
		cv2.destroyAllWindows()
