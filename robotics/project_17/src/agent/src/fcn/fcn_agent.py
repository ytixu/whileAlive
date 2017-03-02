#!/usr/bin/env python
# https://groups.google.com/forum/#!topic/keras-users/EAZJORgWUbI

import threading

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
BATCH_SIZE = 32
smooth = 1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class fcn_agent:
	def __init__(self, data_topic, load_file=None):
		self.data_sub = rospy.Subscriber(data_topic,SensorImages,self.data_update)

		self.bridge = CvBridge()
		self.training_data = [None, None]
		self.compiled = False
		self.trained = False
		self.training = False
		self.sample_n = 0

		if load_file:
			self.model = load_model(load_file)
			self.compiled = True
			self.trained = True

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
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
		conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		# conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
		# conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

		# up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
		# conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
		# conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

		up7 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=1)
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

		self.model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])


		# self.model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
		###'MSE' -- >  'binary_crossentropy'

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
			prediction = self.model.predict_on_batch(np.array([cv2.split(in_image)]))
			image = np.array(prediction[0][0])
			image_gray = np.zeros([image.shape[0],image.shape[1],1])
			image_gray[:,:,0] = image
			cv2.imshow('Prediction', image_gray)
			cv2.waitKey(1)


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

		print 'trained'
		# self.trained = True
		# finally:
		# 	self.lock.release()
		self.model.save('model.h5')


	def destroy(self):
		cv2.destroyAllWindows()
