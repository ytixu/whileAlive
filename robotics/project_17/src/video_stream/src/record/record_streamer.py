#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

class record_streamer:

	def __init__(self, video_file):
		self.pub = rospy.Publisher('image', Image, queue_size=2)
		self.bridge = CvBridge()
		self.video = cv2.VideoCapture(video_file)

	def publish(self):
		rospy.init_node('record_streamer', anonymous=True)

		rate = rospy.Rate(1) # 10hz
		while self.video.isOpened() and not rospy.is_shutdown():
			ret, frame = self.video.read()
			cv2.imshow("Video", frame)

			try:
				self.pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
			except CvBridgeError as e:
				print(e)

			rate.sleep()


		self.video.release()
		cv2.destroyAllWindows()