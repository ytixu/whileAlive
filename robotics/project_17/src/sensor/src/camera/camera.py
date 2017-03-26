#!/usr/bin/env python

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from sensor.msg import SensorImages
from cv_bridge import CvBridge, CvBridgeError

MIN_MOTION = 333
OBS_BOX = 3
SPIN_RATE = 0.33

BACKGROUND_acc_weight = 0.01
SKIP_GROUND = 10
GABOR_SIZE = 12
BLUR_SIZE = 11
N_GABORS = 55

class camera:

	def __init__(self):
		# self.image_sub = rospy.Subscriber(subTopic, Image, self.motion_callback)
		self.bridge = CvBridge()
		self.camera_pub = rospy.Publisher('camera_pub', SensorImages, queue_size=10)


		self.background_avg = None
		self.lastImage = None
		self.gabor_filters = None
		self.cap = None

		rate = rospy.Rate(15) # 10hz
		while not rospy.is_shutdown():
			self.spin()
			rate.sleep()

	def getNextFrame(self):
		if self.cap == None:
			self.cap = cv2.VideoCapture('/home/ytixu/gitHTML/whileAlive/robotics/ob1.avi')

		if type(self.cap) != type(1) and self.cap.isOpened():
			return self.cap.read()
		else:
			self.cap.release()
			self.cap = 1

		return None, None


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

	def withGaborFilter(self, new_image, background_image):
		if self.gabor_filters == None:
			filters = {}
			n = 0
			ksize = GABOR_SIZE
			for theta in np.arange(0, np.pi, np.pi / N_GABORS):
				kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
				kern /= 1.5*kern.sum()
				filters[n] = kern
				n += 1
			self.gabor_filters = filters.values()

		response1 = None
		response2 = None
		image1 = background_image.copy()
		image1 = cv2.copyMakeBorder(image1, top=6, bottom=6, left=6,
										right=6, borderType=cv2.BORDER_REPLICATE)
		image2 = image1.copy()
		image2[6:-6, 6:-6, :] = new_image
		# image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
		image1 = cv2.GaussianBlur(image1, (BLUR_SIZE, BLUR_SIZE), 0)
		# image2 = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
		image2 = cv2.GaussianBlur(image2, (BLUR_SIZE, BLUR_SIZE), 0)
		# cv2.imshow("image1", image1)_
		# cv2.imshow("image2", image2)
		# cv2.imshow("1", image1)
		# cv2.imshow("2", image2)

		for kern in self.gabor_filters:
			img1 = cv2.filter2D(image1, cv2.CV_8UC3, kern)
			img2 = cv2.filter2D(image2, cv2.CV_8UC3, kern)

			if response1 == None:
				response1 = img1
				response2 = img2
			else:
				response1 = np.maximum(response1, img1)
				response2 = np.maximum(response2, img2)

		mean = np.mean(response2)
		std = np.std(response2)
		cut = min(max(mean+1.7*std, 150), 200)
		print mean, std, cut
		response1 = cv2.threshold(response1, cut, 255, cv2.THRESH_BINARY)[1]
		response2 = cv2.threshold(response2, cut, 255, cv2.THRESH_BINARY)[1]
		response = np.abs(np.subtract(response2, response1))
		response = cv2.GaussianBlur(response, (BLUR_SIZE, BLUR_SIZE), 0)
		response = cv2.threshold(response, std, 255, cv2.THRESH_BINARY)[1]
		# cv2.imshow("response", response)
		# cv2.waitKey(1)

		return response[6:-6, 6:-6, :]

	def getSegments(self, image, motion_image):
		cnts, _ = cv2.findContours(motion_image, cv2.RETR_TREE,
									cv2.CHAIN_APPROX_SIMPLE)
		# just for visualization
		final = np.zeros(image.shape, np.uint8)
		mask = np.zeros(motion_image.shape, np.uint8)
		for i,_ in enumerate(cnts):
			mask[...] = 0
			cv2.drawContours(mask, cnts, i, 255, -1)
			# color = color_map(cv2.mean(image, mask))
			color = cv2.mean(image, mask)
			cv2.drawContours(final, cnts, i, color, -1)

		return (cnts, final)

	def spin(self):
		ret, frame = self.getNextFrame()
		cv_image = cv2.resize(frame, None, fx=0.25, fy=0.25)
		motion_image = None
		background = self.background(cv_image)
		cv2.imshow("frame", cv_image)
		cv2.waitKey(1)

		if self.image_diff(cv_image) != None:
			motion_image = self.withGaborFilter(cv_image, background)

		if motion_image != None:
			cv2.imshow("Background", background)
			cv2.imshow("Motion", motion_image)
			# cv2.imshow("cv_image", cv_image)
			motion_image = cv2.cvtColor(motion_image, cv2.COLOR_BGR2GRAY)
			segments, seg_viz = self.getSegments(cv_image, motion_image)
			cv2.imshow("segments", seg_viz)
			cv2.waitKey(1)

			try:
				motion_pub = self.bridge.cv2_to_imgmsg(motion_image, "mono8")
				seg_viz = self.bridge.cv2_to_imgmsg(seg_viz, "bgr8")
				data = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
			except CvBridgeError as e:
				print(e)

			msg = SensorImages()
			msg.input = data
			msg.motion = motion_pub
			msg.segment_viz = seg_viz
			self.camera_pub.publish(msg)

	def destroy(self):
		cv2.destroyAllWindows()
