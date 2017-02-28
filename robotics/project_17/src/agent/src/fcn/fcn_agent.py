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

class fcn_agent:
	def __init__(self, in_topic, out_topic):
		self.input_data_sub = rospy.Subscriber(in_topic,Image,self.input_update)
		self.output_data_sub = rospy.Subscriber(out_topic,Image,self.update)
		self.bridge = CvBridge()
		self.input_data = None
		self.compiled = False

	def _build_model(self, shape):

		self.model = Sequential()
		print shape
		self.model.add(Convolution3D(input_shape=(3,1,shape[1],shape[0]),
								nb_filter=32,
								kernel_dim1=6, kernel_dim2=6, kernel_dim3=6,
								init='uniform',
								activation='relu',
								bias=True,
								border_mode='valid'))   ###to keep the after-convoluted images have same size with original input, you should use 'same'

		self.model.add(MaxPooling3D( pool_size=(2,2,2),
								strides=None,
								border_mode='valid'))

		self.model.add(Convolution3D(nb_filter=32,
								kernel_dim1=4, kernel_dim2=4, kernel_dim3=4,
								init='uniform',
								activation='relu',
								bias=True,
								border_mode='valid'))

		self.model.add(MaxPooling3D( pool_size=(2,2,2),
								strides=None,
								border_mode='valid'))

		self.model.add(Convolution3D(nb_filter=32,
								kernel_dim1=3, kernel_dim2=3, kernel_dim3=3,
								init='uniform',
								activation='relu',
								bias=True,
								border_mode='valid'))

		self.model.add(Convolution3D(nb_filter=32,
								kernel_dim1=3, kernel_dim2=3, kernel_dim3=3,
								init='uniform',
								activation='relu',
								bias=True,
								border_mode='valid'))

		self.model.add(Convolution3D(nb_filter=N_CLASSES,
								kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
								init='uniform',
								activation='linear',     ###linear should be 'sigmoid'
								bias=True,
								border_mode='valid'))


		self.model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
		###'MSE' -- >  'binary_crossentropy'

		self.compiled = True


	def input_update(self, data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		self.input_data = cv_image

		if not self.compiled:
			self._build_model(cv_image.shape)
			return

		prediction = self.model.predict_on_batch(cv_image)
		cv2.imshow('Prediction', prediction)
		cv2.waitKey(1)

	def update(self, data):
		if not self.compiled:
			return

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		self.model.train_on_batch([[self.input_data]], [[cv_image]])

	def destroy(self):
		cv2.destroyAllWindows()
