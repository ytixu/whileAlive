#!/usr/bin/env python

import rospy
from stf_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

class record_streamer:

	def __init__(self, video_file):
		self.pub = rospy.Publisher('image', Image)
		self.bridge = CvBridge()
		self.video = cv2.VideoCapture(video_file)

	def publish(self):
		rate = rospy.Rate(10) # 10hz
		while self.video.isOpened() and rospy.is_shutdown():
    		ret, frame = self.video.read()
    		cv2.imshow("Image window", frame)
			cv2.waitKey(3)

			try:
				self.pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
			except CvBridgeError as e:
				print(e)

			rate.sleep()


		self.video.release()
		cv2.destroyAllWindows()

if __name__ == '__main__':
		streamer = record_streamer('~/gitHTML/whileAlive/robotics/capture.avi')
	try:
		streamer.publish()
	except rospy.ROSInterruptException:
		pass
