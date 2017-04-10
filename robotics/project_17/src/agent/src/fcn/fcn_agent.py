#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor.msg import SensorImages
from cv_bridge import CvBridge, CvBridgeError
from math import log

from mi import mutual_information, entropy


SEG_DECAY = 0.99
NEW_SEG_ALPHA = 1
MIN_MOTION = 111
MIN_SCORE = -10000
MIN_PROB_DIFF = 0.5

thr = 0.000000001

def color_map(color):
	new_c = [0,0,0]
	for i, c in enumerate(color):
		if i > 2 : break
		new_c[i] = int(255 - (int(255-c)/80)*80)
	return (new_c[0], new_c[1], new_c[2], 0)


class Segment:

	def __init__(self, mask=None, raw_image=None, bounding_box=None, labels=None):
		self.mask = mask # binary image

		if raw_image != None:
			self.raw = cv2.bitwise_and(raw_image, raw_image, mask=mask)
			self.entropy = entropy(self.raw)
		else:
			self.raw = None
			self.entropy = 0

		self.weight = 1
		self.labels = None
		self.bounding_box = None
		self.expLabel = None
		self.motion_entropy = None

		if labels != None:
			self.setLabels(labels)
		if bounding_box != None:
			self.updateBoundingBox(bounding_box)


	def updateMask(self, mask):
		if self.mask == None:
			self.mask = mask.copy()
		else:
			self.mask = cv2.bitwise_or(mask, self.mask)

	def updateBoundingBox(self, coord=None, extCoord=None):
		if coord != None:
			x,y,w,h = coord
			extCoord = (x,y,x+w,y+h,w,h)
		if self.bounding_box != None:
			x,y = np.minimum(extCoord[:2], self.bounding_box[:2])
			w,h = np.maximum(extCoord[2:4], self.bounding_box[2:4])
			self.bounding_box = (x,y,w,h,w-x,h-y)
		else:
			self.bounding_box = extCoord

	def setLabels(self, labels):
		self.labels = labels
		self.expLabel = {labels[l]:l for l in labels}[max(labels.values())]

	def getRaw(self, raw):
		if self.mask != None:
			self.raw = cv2.bitwise_and(raw, raw, mask=self.mask)
			self.entropy = entropy(self.raw)
		else:
			shape = raw.shape
			self.mask = np.zeros(shape[:2], np.uint8)
			self.raw = self.mask.copy()

	def label(self):
		return self.expLabel

	def normalizeLabels(self):
		Z = np.sum(self.labels.values())
		self.labels = {l:p/Z for l,p in self.labels.iteritems()}

	def _getBoundedMask(self, bound):
		bmask = np.zeros((bound[5], bound[4]), np.uint8)
		x = min(self.mask.shape[1],bound[2])
		y = min(self.mask.shape[0],bound[3])
		bmask[:y-bound[1],:x-bound[0]] = self.mask[bound[1]:y,bound[0]:x]
		return bmask

	def _agreedBounds(self, segment):
		sbox = segment.bounding_box
		score = np.sum([abs(x-sbox[i])/2.0/(x+sbox[i]) for i,x in enumerate(self.bounding_box)])/6.0
		# score *= self.mask.shape[1]
		# x,y,k,l,w,h = self.bounding_box
		# sx,sy,sk,sl,sw,sh = sbox
		# if w < sw:
		# 	k += sw-w
		# 	w = sw
		# else:
		# 	sk += w-sw
		# 	sw = w
		# if h < sh:
		# 	l += sh-h
		# 	h = sh
		# else:
		# 	sl += h-sh
		# 	sh = h

		# return ((x,y,k,l,w,h), (sx,sy,sk,sl,sw,sh), score)
		return score


	def compareMask(self, segment):
		score = self._agreedBounds(segment)
		# box, sbox, score = self._agreedBounds(segment)
		# print 'mask score', score
		# bounded_mask = self._getBoundedMask(box)
		# sbounded_mask = segment._getBoundedMask(sbox)
		maskInter = cv2.bitwise_and(self.mask, segment.mask)
		# maskInter = cv2.bitwise_and(bounded_mask, sbounded_mask)
		# return np.sum(maskInter)*score/np.sum(bounded_mask)
		return np.sum(maskInter)*(1-score)/np.sum(self.mask)


	def setMotionEntropy(self, entropy):
		self.motion_entropy = entropy

	def decay(self):
		self.weight *= SEG_DECAY


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
		cnts, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for c in cnts:
			self.estimate_segments[0].updateBoundingBox(cv2.boundingRect(c))

		self.color_map[0] = cv2.mean(self.in_image_raw, seg)
		self.estimate_segments[0].setLabels({0:NEW_SEG_ALPHA})

	def getMotionEntropy(self, seg, motion):
		mh = seg.motion_entropy
		if mh == None:
			mh = mutual_information(seg.raw, seg.entropy, motion.raw, motion.entropy)
			mh = mh / seg.entropy[1]
			seg.setMotionEntropy(mh)
		return mh


	def computeProb(self, newSeg, oldSegs, motion):
		# interProp = 1
		# 	interProp = newSeg.compareMask(oldSegs)
		# 	print 'qweqwe', interProp
		# 	if interProp < 0.5:
		# 		# maskInter = cv2.bitwise_or(newSeg.mask, oldSegs.mask)
		# 		# interProp = np.sum(maskInter)*1.0/np.sum(motion.mask)
		# 		# if interProp == 0:
		# 		return {}
		# 		# print 'motion', interProp
		# 	# else:
		# 		# print 'static', interProp

		if type(oldSegs) == type(newSeg):
			oldSegs = {oldSegs.label(): oldSegs}

		label_probs = {}

		mh = self.getMotionEntropy(newSeg, motion)

		for label, oldSeg in oldSegs.iteritems():
			if oldSeg.weight < 0.1:
				continue
			nh = mutual_information(newSeg.raw, newSeg.entropy, oldSeg.raw, oldSeg.entropy)
			# print 'nh-old', nh, newSeg.entropy[1]
			nh = nh / newSeg.entropy[1]
			# print 'nh', nh
			# if nh < 0.05:
			# 	print 'small', nh
			# 	continue
			# cm = newSeg.compareMask(oldSeg)
			# if cm < thr:
			# 	cm = thr

			# if motion.entropy != 0:
				# mutual_information(motion.raw, motion.entropy, oldSeg.raw, oldSeg.entropy)
			moh = self.getMotionEntropy(oldSeg, motion)
			# mh = newSeg.motion_entropy
			# if moh == None:
			# 	# moh = oldSeg.compareMask(motion)
			# 	moh = mutual_information(oldSeg.raw, oldSeg.entropy, motion.raw, motion.entropy)
			# 	moh = moh / oldSeg.entropy[1]
			# 	oldSeg.setMotionEntropy(moh)
			# if abs(mh) < thr:
			# 	if abs(moh) < thr:
			# 		mh = 1
			# 		moh = 1
			# 	else:
			# 		continue
			# elif abs(moh) < thr:
			# 	# cv2.imshow('motion mask', motion.mask)
			# 	# cv2.imshow('seg mask', oldSeg.mask)
			# 	# cv2.waitKey(1)
			# 	# print moh[3]
			# 	continue
			# im = log(interProp) + log(nh)
			mmh = abs(log(1- min(mh/moh, moh/mh)))
			im = mmh * nh
			print label, mh, moh, nh, mmh, '=', im
			# else:
			# 	im = nh

			im *= oldSeg.weight
			print im

			for label, p in oldSeg.labels.iteritems():
				prob = im * p
				if label in label_probs:
					label_probs[label] += prob
				else:
					label_probs[label] = prob

		return label_probs

	def matchSegments(self, new_segments, motion):
		mask = np.zeros(motion.mask.shape, np.uint8)
		segs = {}

		zero_segs = np.zeros(motion.mask.shape, np.uint8)
		new_seg_prob = 1.0 - sum([seg.weight for _, seg in self.estimate_segments.iteritems()])*1.0/len(self.color_map)
		print 'new_seg_prob', new_seg_prob

		print len(new_segments), len(self.estimate_segments)
		for i,cnt in enumerate(new_segments):
			label_probs = {}
			mask[...] = 0
			cv2.drawContours(mask, new_segments, i, 255, -1)
			segs[i] = Segment(mask.copy(), self.in_image_raw, cv2.boundingRect(cnt))
			# cv2.imshow('seg mask', mask)
		# 	cv2.waitKey(1)
			label_probs = self.computeProb(segs[i], self.estimate_segments, motion)

			if len(label_probs) == 0:
				print 'no probs---------'
				label_probs[1000000]
				# label_probs[len(self.color_map)] = NEW_SEG_ALPHA
			else:
				var = 1
				if len(label_probs) > 1:
					var = np.std(label_probs.values()) * 100
					print 'var', var
					print label_probs
				label_probs[len(self.color_map)] = new_seg_prob / var
				#normalization
				Z = np.sum(label_probs.values())
				label_probs = {l:p/Z for l,p in label_probs.iteritems()}

			labels = label_probs
			print labels

			segs[i].setLabels(labels)

		# print len(segs)
		return segs

	def mergeSegs(self, new_segs, motion):
		label_probs = {}

		newSeg = Segment()
		for seg in new_segs:
			newSeg.updateMask(seg.mask)
			newSeg.updateBoundingBox(None, seg.bounding_box)

		newSeg.getRaw(self.in_image_raw)

		if label in self.estimate_segments:
			label_probs = self.computeProb(newSeg, self.estimate_segments[label], motion)
			# if abs(label_probs[label] - max(label_probs.vales())) < thr:

			labels = label_probs
			newSeg.setLabels(labels)
		else:
			newSeg.setLabels({label:NEW_SEG_ALPHA})

		self.estimate_segments[label] = newSeg


	def splitSeg(self, new_seg, motion):
		pass

	def updateSegments(self, new_segments, motion):
		updated_segs = {}
		match_segs = {l: [0,None] for l in self.estimate_segments.keys()}
		new_label = len(self.color_map)

		for i, key in enumerate(new_segments):
			# match old seg to new seg
			for l, p in new_segments.labels.iteritems():
				if l in match_segs and p > match_segs[l][0]:
					match_segs[l] = [p, new_segments[key]]
			# max label
			label = new_segments[key].label()
			if label not in self.color_map:
				self.color_map[label] = cv2.mean(self.in_image_raw, new_segments[key].mask)
			if label in updated_segs:
				updated_segs[label] += [new_segments[key]]
			else:
				updated_segs[label] = [new_segments[key]]

		# new_segments = {}
		for label, segs in updated_segs.iteritems():
			if label in match_segs:
				match_segs[label]
				# something...

			if len(segs) > 1:
				self.mergeSegs(segs, self.estimate_segments[label])
			else:
				if label != new_label:
					del(segs[0].labels[new_label])
					segs[0].normalizeLabels()

				self.estimate_segments[label] = segs[0]
				# new_segments[label] = segs[0]

		# normalize segs label prob
		for l, seg in self.estimate_segments.iteritems():
			seg.setMotionEntropy(None)
			seg.decay()
			print seg.labels

			# cv2.imshow('seg %d' % i, new_segments[key].mask)
		# 	cv2.waitKey(1)
		# return updated_segs


	def getSegments(self, seg_image, sigma=0.33):
		image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)
		image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]
		cnts, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		return cnts

	def getMotionSegments(self, motion_image):
		# movedSeg = {}
		# staticSeg = {}
		movedSeg = Segment()
		for label, seg in self.estimate_segments.iteritems():
			intsect = cv2.bitwise_and(seg.mask, motion_image)
			if np.sum(intsect)/np.max(intsect) > MIN_MOTION:
				movedSeg.updateMask(seg.mask)
				movedSeg.updateBoundingBox(None, seg.bounding_box)
			# else:
			# 	staticSeg[label] = seg

		movedSeg.getRaw(self.in_image_raw)
		return movedSeg


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
				movedSeg = self.getMotionSegments(motion_image)
				segments = self.matchSegments(segments, movedSeg)
				self.updateSegments(segments, movedSeg)

				# (cnts, _) = cv2.findContours(moved_segments.copy(), cv2.RETR_EXTERNAL,
				# 	cv2.CHAIN_APPROX_SIMPLE)

				# print mutual_information(self.prev_frame, seg_viz)
				# cv2.imshow('seg', seg_viz)
				cv2.imshow('motion',motion_image)
				cv2.imshow('acc seg', self.viz())
				cv2.waitKey(1)

				# self.prev_frame = seg_viz.copy()
			print 'done'
		except CvBridgeError as e:
			print(e)


	def destroy(self):
		cv2.destroyAllWindows()
