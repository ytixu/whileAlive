#!/usr/bin/env python
# https://groups.google.com/forum/#!topic/keras-users/EAZJORgWUbI

import rospy
import cv2
import numpy as np
from sensor.msg import SensorImages
from cv_bridge import CvBridge, CvBridgeError

from mi import mutual_information, entropy

# from keras.optimizers import Adam
# from keras import backend as K
# from keras.models import Model, load_model
# from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D

DEFAULT_THRESHOLD = 32
MHI_DURATION = 2
MAX_TIME_DELTA = 2
MIN_TIME_DELTA = 1

def color_map(color):
	new_c = [0,0,0]
	for i, c in enumerate(color):
		if i > 2 : break
		new_c[i] = int(255 - (int(255-c)/80)*80)
	return (new_c[0], new_c[1], new_c[2], 0)


class fcn_agent:
	def __init__(self, data_topic):
		self.bridge = CvBridge()
		self.timestamp = 0
		self.mhi = None
		self.mask = None
		self.orient = None
		self.prev_frame = None
		self.shape = None

		self.estimate_segments = {}
		self.color_map = {}

		self.data_sub = rospy.Subscriber(data_topic,SensorImages,self.data_update)

	def renderSegments(self, segments):
		seg_array = {}
		final = np.zeros(self.shape, np.uint8)
		for i,_ in enumerate(segments):
			final[...] = 0
			cv2.drawContours(final, segments, i, 1, -1)
			seg_array[i] = final

		return seg_array

	def matchSegments(self, new_segments, motion):
		new_seg = np.zeros(self.shape, np.uint8)
		labels = {}
		new_label = len(self.estimate_segments)
		for i,_ in enumerate(new_segments):
			new_seg[...] = 0
			max_label = new_label
			max_entropy = 0
			cv2.drawContours(new_seg, new_segments, i, 1, -1)
			for j, old_seg in self.estimate_segments.iteritems():
				h = mutual_information(old_seg, new_seg)
				if h > max_entropy:
					max_entropy = h
					max_label = j
			if max_label in labels:
				labels[max_label] += [i]
			else:
				labels[max_label] = [i]

		return labels

	def updateSegments(self, new_segments, labels):
		seg_array = {}
		final = np.zeros(self.shape, np.uint8)
		for label, segs in labels.iteritems():
			final[...] = 0
			for seg_id in segs:
				cv2.drawContours(final, new_segments, seg_id, 1, -1)
			seg_array[label] = final
		return seg_array

	def getSegments(self, motion_image):
		cnts, _ = cv2.findContours(motion_image, cv2.RETR_TREE,
									cv2.CHAIN_APPROX_SIMPLE)
		return cnts

	def assignmentColor(self, image):
		# final = np.zeros(image.shape, np.uint8)
		# mask = np.zeros(image.shape, np.uint8)
		for i, seg in self.estimate_segments.iteritems():
			# mask[...] = 0
			# cv2.drawContours(mask, self.estimate_segments, i, 255, -1)
			self.color_map[i] = cv2.mean(image,  seg)
			# cv2.drawContours(final, cnts, i, color, -1)

		# return (cnts, final)

	def viz(self):
		final = np.zeros(self.prev_frame.shape, np.uint8)
		mask = np.zeros(self.prev_frame.shape, np.uint8)
		for i, seg in self.estimate_segments.iteritems():
			mask[:,:,0] = self.color_map[i][0]
			mask[:,:,1] = self.color_map[i][1]
			mask[:,:,2] = self.color_map[i][2]
			final = final + cv2.bitwise_and(mask, mask, mask=seg)

		return final

	def data_update(self, data):
		try:
			in_image_raw = self.bridge.imgmsg_to_cv2(data.input, "bgr8")
			motion_image = self.bridge.imgmsg_to_cv2(data.motion, "mono8")
			seg_viz = self.bridge.imgmsg_to_cv2(data.segment_viz, "bgr8")
		except CvBridgeError as e:
			print(e)

		segments = self.getSegments(motion_image)

		if self.prev_frame == None:
			self.shape = motion_image.shape
			self.prev_frame = seg_viz.copy()
			self.estimate_segments = self.renderSegments(segments)
			self.assignmentColor(in_image_raw)
		else:
			motion = self.update_mhi(seg_viz)
			labels = self.matchSegments(segments, motion)
			self.estimate_segments = self.updateSegments(segments, labels)
			self.assignmentColor(in_image_raw)

			# (cnts, _) = cv2.findContours(moved_segments.copy(), cv2.RETR_EXTERNAL,
			# 	cv2.CHAIN_APPROX_SIMPLE)

			# print mutual_information(self.prev_frame, seg_viz)
			cv2.imshow('seg', seg_viz)
			cv2.imshow('acc seg', self.viz())
 			cv2.waitKey(1)

 			self.prev_frame = seg_viz.copy()


	def update_mhi(self, img):
		if self.mhi == None:
			h, w = img.shape[:2]
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
		# masked_data = cv2.bitwise_and(img, img, mask=mg_mask)
		# masked_data = cv2.bitwise_and(img, img, mask=mg_orient)
		# cv2.imshow('raw', img)
		# cv2.imshow('motempl', mg_mask)
		# cv2.imshow('orient', mg_orient)
 	# 	cv2.waitKey(1)
		return mg_orient

	def destroy(self):
		cv2.destroyAllWindows()
