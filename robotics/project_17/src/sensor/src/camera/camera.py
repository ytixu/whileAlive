#!/usr/bin/env python


# Motion detection
#
# Image diff
# http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
#
# Background average
# http://opencvpython.blogspot.ca/2012/07/background-extraction-using-running.html
#
# Background sub
# https://www.hdm-stuttgart.de/~maucher/Python/ComputerVision/html/Tracking.html
#
# Min-cut/max-flow
# http://pmneila.github.io/PyMaxflow/tutorial.html#complex-grids-with-add-grid-edges

import numpy as np
import scipy

import maxflow
import cv2

import rospy
from sensor_msgs.msg import Image
from sensor.msg import SensorImages
from cv_bridge import CvBridge, CvBridgeError

MIN_MOTION = 333
OBS_BOX = 33
SPIN_RATE = 0.33

BACKGROUND_acc_weight = 0.01
BACKGROUND_history = 10
BACKGROUND_nGauss = 5
BACKGROUND_bgThresh = 0.5
BACKGROUND_noise = 10

class camera:

	def __init__(self, subTopic, method=0):
		self.image_sub = rospy.Subscriber(subTopic, Image, self.motion_callback)
		self.bridge = CvBridge()
		self.camera_pub = rospy.Publisher('camera_pub', SensorImages, queue_size=10)
		# self.snapshot_pub = rospy.Publisher('snapshot', Image, queue_size=10)
		# self.h = Header()

		if method == 1:
			# background subtraction
			self.background_avg = None
			self.bgs = cv2.BackgroundSubtractorMOG(BACKGROUND_history,
				BACKGROUND_nGauss,BACKGROUND_bgThresh,BACKGROUND_noise)
		elif method == 2:
			pass
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

			# if has_motion:
			# 	cv2.drawContours(image, cnts, -1, (0, 255, 0), thickness=-1)
			# 	cv2.imshow("Contour", image)

			self.lastImage = gray

			if has_motion:
				return thresh

		return None

	def background(self, image):
		if self.background_avg == None:
			self.background_avg = np.float32(image)
		else:
			cv2.accumulateWeighted(image,self.background_avg,BACKGROUND_acc_weight)
			background = cv2.convertScaleAbs(self.background_avg)
			return background

		return None

	def background_subtract(self, image):
		foremat=self.bgs.apply(image, learningRate=BACKGROUND_acc_weight)
		# foremat=self.bgs.apply(image)
		# ret,thresh = cv2.threshold(foremat,127,255,0)
		# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		# if len(contours) > 0:
		# 	m= np.mean(contours[0],axis=0)
		return foremat

	def mincut_maxflow(self, image):
		# Create the graph.
		g = maxflow.Graph[int]()
		# Add the nodes. nodeids has the identifiers of the nodes in the grid.
		nodeids = g.add_grid_nodes(image.shape)
		# Add non-terminal edges with the same capacity.
		g.add_grid_edges(nodeids, 50)
		# Add the terminal edges. The image pixels are the capacities
		# of the edges from the source node. The inverted image pixels
		# are the capacities of the edges to the sink node.
		g.add_grid_tedges(nodeids, image, 255-image)
		# Find the maximum flow.
		g.maxflow()
		# Get the segments of the nodes in the grid.
		sgm = g.get_grid_segments(nodeids)
		# The labels should be 1 where sgm is False and 0 otherwise.
		img2 = np.logical_not(sgm) * image
		return img2

	def motion_callback(self, data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		motion_image = None

		if self.method == 1:
			motion_image = self.background_subtract(cv_image)
			# background = self.background(cv_image)
			# motion_image = cv_image - background
		elif self.method == 2:
			motion_image = self.mincut_maxflow(cv_image)
		else:
			motion_image = self.image_diff(cv_image)

		if motion_image != None:
			cv2.imshow("Motion", motion_image)
			cv2.waitKey(1)

			try:
				motion_pub = self.bridge.cv2_to_imgmsg(motion_image, "mono8")
			except CvBridgeError as e:
				print(e)

			msg = SensorImages()
			msg.input = data
			msg.motion = motion_pub
			self.camera_pub.publish(msg)



	def destroy(self):
		cv2.destroyAllWindows()
