#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import keras
from rl.core import Processor


class dqn_segmentation(Processor):
    def process_observation(self, observation):
        # assert observation.ndim == 3  # (height, width, channel)
        # img = Image.fromarray(observation)
        # img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        # processed_observation = np.array(img)
        # assert processed_observation.shape == INPUT_SHAPE
        # return processed_observation.astype('uint8')  # saves storage in experience memory
        return observation

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        # processed_batch = batch.astype('float32') / 255.
        return batch

    def process_reward(self, reward):
        # return np.clip(reward, -1., 1.)
        return reward