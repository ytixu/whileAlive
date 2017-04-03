#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor.msg import SensorImages
from cv_bridge import CvBridge, CvBridgeError
from math import log

from mi import mutual_information, entropy


SEG_DECRAY = 0.1
NEW_SEG_ALPHA = 1

def color_map(color):
	new_c = [0,0,0]
	for i, c in enumerate(color):
		if i > 2 : break
		new_c[i] = int(255 - (int(255-c)/80)*80)
	return (new_c[0], new_c[1], new_c[2], 0)


class Segment:

	def __init__(self, mask, raw_image, labels=None):
		self.mask = mask # binary image
		self.raw = cv2.bitwise_and(raw_image, raw_image, mask=mask)
		self.entropy = entropy(self.raw)
		self.labels = None
		self.expLabel = None
		if labels != None:
			self.setLabels(labels)

	def setLabels(self, labels):
		self.labels = labels
		self.expLabel = self.labels[max(self.labels.keys())]

	def label(self):
		return self.expLabel


class fcn_agent:
	def __init__(self, data_topic):
		self.bridge = CvBridge()
		self.shape = None

		self.estimate_segments = {}
		self.color_map = {}
		self.in_image_raw = None

		self.data_sub = rospy.Subscriber(data_topic,SensorImages,self.data_update)

	def renderSegments(self, motion_image):
		seg = cv2.cvtColor(motion_image, cv2.COLOR_BGR2GRAY)
		self.estimate_segments[0] = Segment(seg, self.in_image_raw)
		self.color_map[0] = cv2.mean(self.in_image_raw, seg)
		self.estimate_segments[0].setLabels({NEW_SEG_ALPHA:0})

	def matchSegments(self, new_segments, motion):
		mask = np.zeros(motion.shape, np.uint8)
		segs = {}
		motion_raw = cv2.bitwise_and(self.in_image_raw, self.in_image_raw, mask=motion)
		motion_entropy = entropy(motion_raw)

		zero_segs = np.zeros(motion.shape, np.uint8)
		has_zero_segs = False

		print len(new_segments), len(self.estimate_segments)
		for i,_ in enumerate(new_segments):
			label_probs = {}
			mask[...] = 0
			cv2.drawContours(mask, new_segments, i, 255, -1)
			segs[i] = Segment(mask.copy(), self.in_image_raw)
			# cv2.imshow('seg mask', mask)
	 	# 	cv2.waitKey(1)
			for j, seg in self.estimate_segments.iteritems():
				cp = mutual_information(segs[i].raw, segs[i].entropy, seg.raw, seg.entropy)
				mp = mutual_information(motion_raw, motion_entropy, seg.raw, seg.entropy)
				mcp = mutual_information(segs[i].raw, segs[i].entropy, motion_raw, motion_entropy)
				im = log(cp) + log(mp) + log(mcp)
				print cp, im
				if cp < 0.05: continue
				# cv2.imshow('seg', segs[i].raw)
				# cv2.imshow('old seg', seg.raw)
	 		# 	cv2.waitKey(1)
				for p, label in seg.labels.iteritems():
					prob = im + p
					if label in label_probs:
						label_probs[label] += prob
					else:
						label_probs[label] = prob

			labels = {p: l for l, p in label_probs.iteritems()}

			if len(labels) == 0:
				print 'no seg'
				cv2.drawContours(zero_segs, new_segments, i, 255, -1)
				has_zero_segs = i
				del(segs[i])
			else:
				segs[i].setLabels(labels)

		if has_zero_segs != False:
			labels = {NEW_SEG_ALPHA:len(self.color_map)}
			segs[has_zero_segs] = Segment(zero_segs, self.in_image_raw, labels)

		print len(segs)
		return segs


	def updateSegments(self, new_segments, motion):
		updated_segs = {}
		zero_segs = np.zeros(motion.shape, np.uint8)
		for i, key in enumerate(new_segments):
			label = new_segments[key].label()
			if label not in self.color_map:
				self.color_map[label] = cv2.mean(self.in_image_raw, new_segments[key].mask)
			updated_segs[i] = new_segments[key]
			# cv2.imshow('seg %d' % i, new_segments[key].mask)
 		# 	cv2.waitKey(1)
		return updated_segs


	def getSegments(self, motion_image, sigma=0.33):
		image = cv2.cvtColor(motion_image, cv2.COLOR_BGR2GRAY)
		image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]
		cnts, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		return cnts

	def assignmentColor(self):
		# final = np.zeros(image.shape, np.uint8)
		# mask = np.zeros(image.shape, np.uint8)
		for i, seg in self.estimate_segments.iteritems():
			# mask[...] = 0
			# cv2.drawContours(mask, self.estimate_segments, i, 255, -1)
			self.color_map[i] = cv2.mean(self.in_image_raw, seg.mask)
			# cv2.imshow('seg', segs[i].raw)
			# cv2.drawContours(final, cnts, i, color, -1)

		# return (cnts, final)

	def viz(self):
		final = np.zeros(self.shape, np.uint8)
		mask = np.zeros(self.shape, np.uint8)
		for i, seg in self.estimate_segments.iteritems():
			label = seg.label()
			mask[:,:,0] = self.color_map[label][0]
			mask[:,:,1] = self.color_map[label][1]
			mask[:,:,2] = self.color_map[label][2]
			new = cv2.bitwise_and(mask, mask, mask=seg.mask)
			final = final + new

		return final

	def data_update(self, data):
		print 'start'
		try:
			self.in_image_raw = self.bridge.imgmsg_to_cv2(data.input, "bgr8")
			motion_image = self.bridge.imgmsg_to_cv2(data.motion, "mono8")
			seg_viz = self.bridge.imgmsg_to_cv2(data.segment_viz, "bgr8")

			if self.shape == None:
				self.shape = seg_viz.shape
				self.renderSegments(seg_viz)
				self.assignmentColor()
			else:
				segments = self.getSegments(seg_viz)
				segments = self.matchSegments(segments, motion_image)
				self.estimate_segments = self.updateSegments(segments, motion_image)

				# (cnts, _) = cv2.findContours(moved_segments.copy(), cv2.RETR_EXTERNAL,
				# 	cv2.CHAIN_APPROX_SIMPLE)

				# print mutual_information(self.prev_frame, seg_viz)
				# cv2.imshow('seg', seg_viz)
				cv2.imshow('acc seg', self.viz())
	 			cv2.waitKey(1)

	 			# self.prev_frame = seg_viz.copy()
	 		print 'done'
		except CvBridgeError as e:
			print(e)


	def destroy(self):
		cv2.destroyAllWindows()
