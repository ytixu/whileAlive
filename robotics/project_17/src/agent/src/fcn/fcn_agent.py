#!/usr/bin/env python
# https://groups.google.com/forum/#!topic/keras-users/EAZJORgWUbI

import threading

import rospy
import cv2
import numpy as np
from sensor.msg import SensorImages
from cv_bridge import CvBridge, CvBridgeError

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D

N_CLASSES = 2
BATCH_SIZE = 32

class fcn_agent:
	def __init__(self, data_topic):
		self.data_sub = rospy.Subscriber(data_topic,SensorImages,self.data_update)

		self.bridge = CvBridge()
		self.training_data = [None, None]
		self.compiled = False
		self.trained = False
		self.sample_n = 0

	def _build_model(self, shape):
		inputs = Input((3, shape[0], shape[1]))
		conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
		conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
		conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
		conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
		# pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		# conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
		# conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
		# pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		# conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
		# conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

		# up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
		# conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
		# conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

		# up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
		# conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
		# conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

		up8 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
		conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
		conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

		up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
		conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
		conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

		conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

		self.model = Model(input=inputs, output=conv10)

		# model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])


		self.model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
		###'MSE' -- >  'binary_crossentropy'

		self.compiled = True
		print 'compiled'


	def data_update(self, data):
		try:
			in_image = self.bridge.imgmsg_to_cv2(data.input, "bgr8")
			out_image = self.bridge.imgmsg_to_cv2(data.motion, "mono8")
		except CvBridgeError as e:
			print(e)

		if self.training_data[1] == None:
			self.training_data[0] = np.array([cv2.split(in_image)])
			self.training_data[1] = np.array([cv2.split(out_image)])
		else:
			self.training_data[0] = np.append(self.training_data[0],
											  np.array([cv2.split(in_image)]),
											  axis=0)
			self.training_data[1] = np.append(self.training_data[1],
											  np.array([cv2.split(out_image)]),
											  axis=0)
		self.sample_n += 1
		if self.sample_n > BATCH_SIZE:
			self.update()

		if not self.compiled:
			self._build_model(in_image.shape)
			return

		if self.trained:
			prediction = self.model.predict_on_batch([in_image])
			print prediction
			cv2.imshow('Prediction', prediction)
			cv2.waitKey(1)


	def update(self):
		# self.lock.acquire()
		# try:
		print 'training'
		x = self.training_data[0]
		y = self.training_data[1]

		self.training_data = [None, None]
		self.sample_n = 0

		self.model.fit(x, y, verbose=1)

		print 'trained'
		# self.trained = True
		# finally:
		# 	self.lock.release()


	def destroy(self):
		cv2.destroyAllWindows()
