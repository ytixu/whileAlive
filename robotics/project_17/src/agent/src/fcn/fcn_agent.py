#!/usr/bin/env python
# https://groups.google.com/forum/#!topic/keras-users/EAZJORgWUbI

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from keras.models import Sequential
from keras.layers.convolutional import Convolution3D, MaxPooling3D

N_CLASSES = 2
BATCH_SIZE = 10

class fcn_agent:
	def __init__(self, in_topic, out_topic):
		self.input_data_sub = rospy.Subscriber(in_topic,Image,self.input_update)
		self.output_data_sub = rospy.Subscriber(out_topic,Image,self.update)
		self.bridge = CvBridge()
		self.input_data = None
		self.training_data = [[[],[],[]],[]]
		self.compiled = False
		self.trained = False

	def _build_model(self, shape):

		self.model = Sequential()
		conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
		conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
		conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
		conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
		conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
		conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

		up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
		conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
		conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

		up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
		conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
		conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

		up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
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


	def input_update(self, data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		self.input_data = cv_image

		if not self.compiled:
			self._build_model(cv_image.shape)
			return

		if self.trained:
			prediction = self.model.predict_on_batch([cv_image])
			print prediction
			cv2.imshow('Prediction', prediction)
			cv2.waitKey(1)

	def update(self, data):
		if not self.compiled:
			return

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
		except CvBridgeError as e:
			print(e)

		R,G,B = cv2.split(self.input_data)
		self.training_data[0][0] += [np.transpose(R)]
		self.training_data[0][1] += [np.transpose(G)]
		self.training_data[0][2] += [np.transpose(B)]
		self.training_data[1] += [np.transpose(cv_image)]
		n = len(self.training_data[1])

		if n > BATCH_SIZE:
			print 'training'
			x = np.array([[self.training_data[0][0][:BATCH_SIZE],
									  self.training_data[0][1][:BATCH_SIZE],
									  self.training_data[0][2][:BATCH_SIZE]]])
			y = np.array([self.training_data[1][:BATCH_SIZE]])

			self.training_data[0][0] = self.training_data[0][0][BATCH_SIZE:]
			self.training_data[0][1] = self.training_data[0][1][BATCH_SIZE:]
			self.training_data[0][2] = self.training_data[0][2][BATCH_SIZE:]
			self.training_data[1] = self.training_data[1][BATCH_SIZE:]

			self.model.train_on_batch(x, y)

			print 'trained'
			self.trained = True

	def destroy(self):
		cv2.destroyAllWindows()
