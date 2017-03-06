#!/usr/bin/env python
# https://groups.google.com/forum/#!topic/keras-users/EAZJORgWUbI

import time
import rospy
import cv2
import numpy as np
from sensor.msg import SensorImages
from cv_bridge import CvBridge, CvBridgeError

from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D

N_CLASSES = 2
BATCH_SIZE = 31
SMOOTH = 1

GABOR_FILTERS = None

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def withGaborFilter(image):
	global GABOR_FILTERS

	if GABOR_FILTERS == None:
		filters = {}
		n = 0
		ksize = 16
		for theta in np.arange(0, np.pi, np.pi / 16):
			kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
			kern /= 1.5*kern.sum()
			filters[n] = kern
			n += 1
		GABOR_FILTERS = filters.values()

	responses = None
	for kern in GABOR_FILTERS:
		img = cv2.filter2D(image, cv2.CV_8UC3, kern)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# VISUALIZE GABOR FILTERED INPUT
		# cv2.imshow('gabors', img)
		# cv2.waitKey(1)
		# time.sleep(0.5)

		if responses == None:
			responses = np.array([img])
		else:
			responses = np.append(responses, np.array([img]), axis=0)
	image = image.reshape((image.shape[2], image.shape[0], image.shape[1]))
	return np.append(image, responses, axis=0)


class fcn_agent:
	def __init__(self, data_topic, load_file=None):
		if load_file:
			self.compiled = True
			self.trained = True
			self.model = load_model(load_file,
				custom_objects={'dice_coef_loss' : dice_coef_loss,
								'dice_coef' : dice_coef})
		else:
			self.compiled = False
			self.trained = False

		self.data_sub = rospy.Subscriber(data_topic,SensorImages,self.data_update)

		self.bridge = CvBridge()
		self.training_data = [None, None]
		self.training = False
		self.sample_n = 0


	def _build_model(self, shape):
		inputs = Input(shape)
		# conv1 = Convolution2D(32, 5, 5, activation='sigmoid', border_mode='same')(inputs)
		# conv1 = Convolution2D(32, 5, 5, activation='sigmoid', border_mode='same')(conv1)
		# pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Convolution2D(32, 5, 5, activation='sigmoid', border_mode='same')(inputs)
		conv2 = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Convolution2D(64, 5, 5, activation='sigmoid', border_mode='same')(pool2)
		conv3 = Convolution2D(64, 5, 5, activation='relu', border_mode='same')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Convolution2D(128, 5, 5, activation='sigmoid', border_mode='same')(pool3)
		conv4 = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(conv4)
		# pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		up5 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=1)
		conv5 = Convolution2D(64, 5, 5, activation='sigmoid', border_mode='same')(up5)
		conv5 = Convolution2D(64, 5, 5, activation='relu', border_mode='same')(conv5)

		up6 = merge([UpSampling2D(size=(2, 2))(up5), conv2], mode='concat', concat_axis=1)
		conv6 = Convolution2D(64, 5, 5, activation='sigmoid', border_mode='same')(up6)
		conv6 = Convolution2D(64, 5, 5, activation='relu', border_mode='same')(conv6)

		conv7 = Convolution2D(1, 1, 1, activation='sigmoid')(up6)

		# conv1 = Convolution2D(32, 5, 5, activation='sigmoid', border_mode='same')(inputs)
		# conv1 = Convolution2D(32, 5, 5, activation='sigmoid', border_mode='same')(conv1)
		# pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		# conv2 = Convolution2D(64, 5, 5, activation='sigmoid', border_mode='same')(pool1)
		# conv2 = Convolution2D(64, 5, 5, activation='sigmoid', border_mode='same')(conv2)
		# pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		# conv3 = Convolution2D(128, 5, 5, activation='sigmoid', border_mode='same')(pool2)
		# conv3 = Convolution2D(128, 5, 5, activation='sigmoid', border_mode='same')(conv3)
		# pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		# conv4 = Convolution2D(256, 5, 5, activation='sigmoid', border_mode='same')(pool3)
		# conv4 = Convolution2D(256, 5, 5, activation='sigmoid', border_mode='same')(conv4)
		# pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		# conv5 = Convolution2D(512, 5, 5, activation='sigmoid', border_mode='same')(pool4)
		# conv5 = Convolution2D(512, 5, 5, activation='sigmoid', border_mode='same')(conv5)

		# up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
		# conv6 = Convolution2D(256, 5, 5, activation='sigmoid', border_mode='same')(up6)
		# conv6 = Convolution2D(256, 5, 5, activation='sigmoid', border_mode='same')(conv6)

		# up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
		# conv7 = Convolution2D(128, 5, 5, activation='sigmoid', border_mode='same')(up7)
		# conv7 = Convolution2D(128, 5, 5, activation='sigmoid', border_mode='same')(conv7)

		# up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
		# conv8 = Convolution2D(64, 5, 5, activation='sigmoid', border_mode='same')(up8)
		# conv8 = Convolution2D(64, 5, 5, activation='sigmoid', border_mode='same')(conv8)

		# up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
		# conv9 = Convolution2D(32, 5, 5, activation='sigmoid', border_mode='same')(up9)
		# conv9 = Convolution2D(32, 5, 5, activation='sigmoid', border_mode='same')(conv9)

		# conv10 = Convolution2D(1, 1, 1, activation='relu')(conv9)

		# self.model = Model(input=inputs, output=conv10)
		self.model = Model(input=inputs, output=conv7)

		# self.model.compile(optimizer='sgd', loss=dice_coef_loss, metrics=[dice_coef])
		self.model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

		# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		# self.model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
		###'MSE' -- >  'binary_crossentropy'

		from keras.utils.visualize_util import plot
		plot(self.model, to_file='model.png')

		self.compiled = True
		print 'compiled'


	def data_update(self, data):
		if self.training:
			return
		try:
			in_image = self.bridge.imgmsg_to_cv2(data.input, "bgr8")
			out_image = self.bridge.imgmsg_to_cv2(data.motion, "mono8")
		except CvBridgeError as e:
			print(e)

		# simulate first layers of gabor filters
		_,out_image = cv2.threshold(out_image,127,255,cv2.THRESH_BINARY)
		self.sample_n += 1
		in_image = withGaborFilter(in_image)


		if self.trained:
			prediction = self.model.predict_on_batch(np.array([in_image]))
			image = np.array(prediction[0][0])
			print np.max(image), np.min(image), np.mean(image)
			cv2.imshow('Prediction', image)
			_,image_gray = cv2.threshold(image,0.5,1,cv2.THRESH_BINARY)
			# image = np.zeros([image.shape[0],image.shape[1],1])
			# image[:,:,0] = image
			# image_gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
			cv2.waitKey(1)
			# print dice_coef_loss(out_image, prediction)
		else:
			if self.training_data[1] == None:
				self.training_data[0] = np.array([in_image])
				self.training_data[1] = np.array([cv2.split(out_image)])
			else:
				self.training_data[0] = np.append(self.training_data[0],
												  np.array([in_image]),
												  axis=0)
				self.training_data[1] = np.append(self.training_data[1],
												  np.array([cv2.split(out_image)]),
												  axis=0)

			if not self.trained and self.sample_n > BATCH_SIZE:
				self.update()

			if not self.compiled:
				self._build_model(in_image.shape)
				return




	def update(self):
		# self.lock.acquire()
		# try:
		if self.training:
			return

		self.training= True

		print 'training'
		x = self.training_data[0]
		y = self.training_data[1]

		self.training_data = [None, None]
		self.sample_n = 0

		self.model.fit(x, y, verbose=1)

		# self.trained = True
		# finally:
		# 	self.lock.release()
		self.model.save('model.h5')
		self.trained= True
		self.training = False
		print 'trained'


	def destroy(self):
		cv2.destroyAllWindows()
