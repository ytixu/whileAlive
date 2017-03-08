#!/usr/bin/env python
# https://groups.google.com/forum/#!topic/keras-users/EAZJORgWUbI

import rospy
import cv2
import numpy as np
from sensor.msg import SensorImages
from cv_bridge import CvBridge, CvBridgeError

# from keras.optimizers import Adam
# from keras import backend as K
# from keras.models import Model, load_model
# from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D

DEFAULT_THRESHOLD = 32
MHI_DURATION = 2
MAX_TIME_DELTA = 2
MIN_TIME_DELTA = 1

class fcn_agent:
	def __init__(self, data_topic):
		self.bridge = CvBridge()
		self.timestamp = 0
		self.mhi = None
		self.mask = None
		self.orient = None
		self.prev_frame = None

		self.data_sub = rospy.Subscriber(data_topic,SensorImages,self.data_update)


	def data_update(self, data):
		try:
			in_image_raw = self.bridge.imgmsg_to_cv2(data.input, "bgr8")
			out_image_raw = self.bridge.imgmsg_to_cv2(data.motion, "bgr8")
		except CvBridgeError as e:
			print(e)

		self.update_mhi(out_image_raw)


	def update_mhi(self, img):
		if self.mhi == None:
			h, w = img.shape[:2]
			self.prev_frame = img.copy()
			self.mhi = np.zeros((h, w), np.float32)
			self.mask = np.zeros((h, w), np.float32)
			self.orient = np.zeros((h, w), np.float32)
			return

		frame_diff = cv2.absdiff(img, self.prev_frame)
		gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
		ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
		self.timestamp += 1

		# update motion history
		cv2.updateMotionHistory(fgmask, self.mhi, self.timestamp, MHI_DURATION)

		# normalize motion history
		# cv2.cvtScale(self.mhi, self.mask, 255./MHI_DURATION,(MHI_DURATION - timestamp)*255./MHI_DURATION)
		mg_mask, mg_orient = cv2.calcMotionGradient(self.mhi, MAX_TIME_DELTA, MIN_TIME_DELTA, apertureSize=5)
		mg_orient = cv2.threshold(mg_orient, 1, 255, cv2.THRESH_BINARY)[1]/255
		mg_orient = mg_orient.astype('uint8')
		masked_data = cv2.bitwise_and(img, img, mask=mg_mask)
		masked_data = cv2.bitwise_and(img, img, mask=mg_orient)
		cv2.imshow('motempl', masked_data)
		# cv2.imshow('raw', img)
 		cv2.waitKey(1)
		prev_frame = img.copy()

	def destroy(self):
		cv2.destroyAllWindows()
