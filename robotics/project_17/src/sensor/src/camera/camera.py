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
import time

import maxflow
import cv2

import rospy
from sensor_msgs.msg import Image
from sensor.msg import SensorImages
from cv_bridge import CvBridge, CvBridgeError

MIN_MOTION = 333
OBS_BOX = 3
SPIN_RATE = 0.33

BACKGROUND_acc_weight = 0.01
BACKGROUND_history = 10
BACKGROUND_nGauss = 5
BACKGROUND_bgThresh = 0.5
BACKGROUND_noise = 10
SKIP_GROUND = 10
GABOR_SIZE = 12
BLUR_SIZE = 11

def diffImg(t0, t1, t2):
	d1 = cv2.absdiff(t2, t1)
	d2 = cv2.absdiff(t1, t0)
	return cv2.bitwise_and(d1, d2)

class camera:

	def __init__(self, subTopic, method=0):
		self.image_sub = rospy.Subscriber(subTopic, Image, self.motion_callback)
		self.bridge = CvBridge()
		self.camera_pub = rospy.Publisher('camera_pub', SensorImages, queue_size=10)
		# self.snapshot_pub = rospy.Publisher('snapshot', Image, queue_size=10)
		# self.h = Header()

		if method == 1:
			# background subtraction
			self.skip_ground = 0
			self.bgs = cv2.BackgroundSubtractorMOG(BACKGROUND_history,
				BACKGROUND_nGauss,BACKGROUND_bgThresh,BACKGROUND_noise)
		elif method == 2:
			pass
		else:
			self.background_avg = None
			self.lastImage = None
			self.gabor_filters = None
			self.t0 = None
			self.t1 = None
			self.t2 = None

		self.method = method

	def image_diff(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)

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
		if self.skip_ground == SKIP_GROUND:
			self.skip_ground = 0
			foremat = self.bgs.apply(image, learningRate=BACKGROUND_acc_weight)
			return foremat
		else:
			self.skip_ground += 1
		return None


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

	def withGaborFilter(self, new_image, background_image):
		if self.gabor_filters == None:
			filters = {}
			n = 0
			ksize = GABOR_SIZE
			for theta in np.arange(0, np.pi, np.pi / 64):
				kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
				kern /= 1.5*kern.sum()
				filters[n] = kern
				n += 1
			self.gabor_filters = filters.values()

		response1 = None
		response2 = None
		image1 = cv2.copyMakeBorder(background_image, top=6, bottom=6, left=6,
										right=6, borderType=cv2.BORDER_REPLICATE)
		image2 = image1.copy()
		image2[6:-6, 6:-6, :] = new_image
		# image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
		image1 = cv2.GaussianBlur(image1, (BLUR_SIZE, BLUR_SIZE), 0)
		# image2 = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
		image2 = cv2.GaussianBlur(image2, (BLUR_SIZE, BLUR_SIZE), 0)
		# cv2.imshow("image1", image1)
		# cv2.imshow("image2", image2)

		for kern in self.gabor_filters:
			img1 = cv2.filter2D(image1, cv2.CV_8UC3, kern)
			img2 = cv2.filter2D(image2, cv2.CV_8UC3, kern)

			if response1 == None:
				response1 = img1
				response2 = img2
			else:
				response1 = np.maximum(response1, img1)
				response2 = np.maximum(response2, img2)

		# cv2.imshow("1", response1)
		# cv2.imshow("2", response2)
		mean = np.mean(response2)
		std = np.std(response2)
		print mean, std
		cut = max(150, mean+1.7*std)
		response1 = cv2.threshold(response1, cut, 255, cv2.THRESH_BINARY)[1]
		response2 = cv2.threshold(response2, cut, 255, cv2.THRESH_BINARY)[1]
		response = np.abs(np.subtract(response2, response1))
		response = cv2.GaussianBlur(response, (BLUR_SIZE, BLUR_SIZE), 0)
		response = cv2.threshold(response, std, 255, cv2.THRESH_BINARY)[1]
		# cv2.imshow("response", response)
		# cv2.waitKey(1)

		return response[6:-6, 6:-6, :]

	def getSegments(self, image, motion_image):
		cnts, _ = cv2.findContours(motion_image, cv2.RETR_EXTERNAL,
									cv2.CHAIN_APPROX_SIMPLE)
		final = np.zeros(image.shape, np.uint8)
		mask = np.zeros(motion_image.shape, np.uint8)
		for i,_ in enumerate(cnts):
			mask[...] = 0
			cv2.drawContours(mask, cnts, i, 255, -1)
			cv2.drawContours(final, cnts, i, cv2.mean(image, mask), -1)

		return final

	def getImageGradient(self, currentImage):
		if self.t2 == None:
			if self.t1 == None:
				if self.t0 == None:
					self.t0 = currentImage
				else:
					self.t1 = currentImage
			else:
				self.t2 = currentImage
		else:
			self.t0 = self.t1
			self.t1 = self.t2
			self.t2 = currentImage

		return diffImg(self.t0 , self.t1 , self.t2)

	def motion_callback(self, data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		motion_image = None
		background = self.background(cv_image)

		if self.method == 1:
			motion_image = self.background_subtract(cv_image)
			# motion_image = cv_image - background
		elif self.method == 2:
			motion_image = self.mincut_maxflow(cv_image)

		elif self.image_diff(cv_image) != None:
			motion_image = self.withGaborFilter(cv_image, background)

		if motion_image != None:
			gradient = self.getImageGradient(cv_image)
			print gradient

			cv2.imshow("Motion", gradient)
			cv2.imshow("cv_image", cv_image)
			motion_image = cv2.cvtColor(motion_image, cv2.COLOR_BGR2GRAY)
			segments = self.getSegments(cv_image, motion_image)
			cv2.imshow("segments", segments)
			cv2.waitKey(1)

			try:
				motion_pub = self.bridge.cv2_to_imgmsg(segments, "bgr8")
			except CvBridgeError as e:
				print(e)

			msg = SensorImages()
			msg.input = data
			msg.motion = motion_pub
			self.camera_pub.publish(msg)

	def destroy(self):
		cv2.destroyAllWindows()
