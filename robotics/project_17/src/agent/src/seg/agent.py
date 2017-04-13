import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from math import log, isnan

MIN_MOTION = 222

BACKGROUND_acc_weight = 0.01
GABOR_SIZE = 12
BLUR_SIZE = 11
N_GABORS = 55

class Segment:

	def __init__(self, mask=None, raw_image=None, bounding_box=None, labels=None):
		self.mask = mask # binary image

		if raw_image != None:
			self.raw = cv2.bitwise_and(raw_image, raw_image, mask=mask)
			self.entropy = entropy(self.raw)
		else:
			self.raw = None
			self.entropy = 0

		self.weight = 1.0
		self.labels = None
		self.bounding_box = None
		self.expLabel = None
		self.motion_entropy = None

		if labels != None:
			self.setLabels(labels)
		if bounding_box != None:
			self.updateBoundingBox(bounding_box)


	def updateMask(self, mask, add=1):
		if self.mask == None:
			self.mask = mask.copy()
		elif add > 0:
			self.mask = cv2.bitwise_or(mask, self.mask)
		elif add < 0:
			self.mask = cv2.bitwise_and(self.mask, self.mask, mask=cv2.bitwise_not(mask))
		else:
			self.mask = mask

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

	def setMotionEntropy(self, entropy):
		self.motion_entropy = entropy

	def decay(self):
		self.weight *= SEG_DECAY


class seg_agent:
	def __init__(self, data_topic):
		self.bridge = CvBridge()

		self.gabor_filters = None
		self.background_avg = None
		self.estimate_segments = {}
		self.color_map = {}
		self.in_image_raw = {}
		self.in_image_raw_prev = {}

		self.data_sub = rospy.Subscriber(data_topic,Image,self.data_update)
		self.min_movement = 1000

	def optic_flow(self, small_gray):
		flow = cv2.calcOpticalFlowFarneback(self.in_image_raw_prev[0], small_gray, 0.5, 1, 3, 15, 3, 5, 1)
		flow_arr = np.array(flow)
		sh = flow_arr.shape
		flow_img = np.asarray(np.reshape(np.linalg.norm(np.reshape(flow_arr, (sh[0] * sh[1], sh[2])), axis=1), (sh[0], sh[1])), np.uint8)
		mean = np.mean(flow_img)
		std = np.std(flow_img)
		if self.min_movement > mean:
			self.min_movement = mean

		thresh = cv2.threshold(flow_img, self.min_movement+std, 255, cv2.THRESH_BINARY)[1]
		(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)

		has_motion = False
		motion = np.zeros(thresh.shape, np.uint8)
		for i, c in enumerate(cnts):
			if cv2.contourArea(c) < MIN_MOTION:
				continue
			cv2.drawContours(motion, cnts, i, 255, -1)
			has_motion = True


		if has_motion:
			return motion

		return None


	def background(self, image):
		has_acc = True
		if self.background_avg == None:
			self.background_avg = np.float32(image)
			has_acc = False

		cv2.accumulateWeighted(image,self.background_avg,BACKGROUND_acc_weight)
		if not has_acc:
			return None

		background = cv2.convertScaleAbs(self.background_avg)
		return background

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
		image1 = cv2.GaussianBlur(image1, (BLUR_SIZE, BLUR_SIZE), 0)
		image2 = cv2.GaussianBlur(image2, (BLUR_SIZE, BLUR_SIZE), 0)

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
		resp1 = cv2.threshold(response1, cut, 255, cv2.THRESH_BINARY)[1]
		resp2 = cv2.threshold(response2, cut, 255, cv2.THRESH_BINARY)[1]
		response = np.abs(np.subtract(resp2, resp1))
		response = cv2.GaussianBlur(response, (BLUR_SIZE, BLUR_SIZE), 0)
		response = cv2.threshold(response, std, 255, cv2.THRESH_BINARY)[1]

		return (response[6:-6, 6:-6, :], response2[6:-6, 6:-6, :])


	def data_update(self, data):
		print 'start'
		self.in_image_raw_prev = self.in_image_raw

		try:
			frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)
			return False

		small_frame = cv2.resize(frame, None, fx=0.25, fy=0.25)
		small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
		self.in_image_raw = [small_gray, small_frame, frame]

		cv2.imshow("frame", small_frame)
		cv2.waitKey(1)

		background = self.background(small_frame)

		if background == None:
			return True

		cv2.imshow("background", background)
		cv2.waitKey(1)

		motion = self.optic_flow(small_gray)
		if motion != None:
			cv2.imshow("motion", motion)
			cv2.waitKey(1)


		# if motion != None:
		# 	motion_image, filtered_image = self.withGaborFilter(cv_image, background)

		# if motion_image != None: