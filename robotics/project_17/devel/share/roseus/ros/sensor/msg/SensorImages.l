;; Auto-generated. Do not edit!


(when (boundp 'sensor::SensorImages)
  (if (not (find-package "SENSOR"))
    (make-package "SENSOR"))
  (shadow 'SensorImages (find-package "SENSOR")))
(unless (find-package "SENSOR::SENSORIMAGES")
  (make-package "SENSOR::SENSORIMAGES"))

(in-package "ROS")
;;//! \htmlinclude SensorImages.msg.html
(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))


(defclass sensor::SensorImages
  :super ros::object
  :slots (_input _motion ))

(defmethod sensor::SensorImages
  (:init
   (&key
    ((:input __input) (instance sensor_msgs::Image :init))
    ((:motion __motion) (instance sensor_msgs::Image :init))
    )
   (send-super :init)
   (setq _input __input)
   (setq _motion __motion)
   self)
  (:input
   (&rest __input)
   (if (keywordp (car __input))
       (send* _input __input)
     (progn
       (if __input (setq _input (car __input)))
       _input)))
  (:motion
   (&rest __motion)
   (if (keywordp (car __motion))
       (send* _motion __motion)
     (progn
       (if __motion (setq _motion (car __motion)))
       _motion)))
  (:serialization-length
   ()
   (+
    ;; sensor_msgs/Image _input
    (send _input :serialization-length)
    ;; sensor_msgs/Image _motion
    (send _motion :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; sensor_msgs/Image _input
       (send _input :serialize s)
     ;; sensor_msgs/Image _motion
       (send _motion :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; sensor_msgs/Image _input
     (send _input :deserialize buf ptr-) (incf ptr- (send _input :serialization-length))
   ;; sensor_msgs/Image _motion
     (send _motion :deserialize buf ptr-) (incf ptr- (send _motion :serialization-length))
   ;;
   self)
  )

(setf (get sensor::SensorImages :md5sum-) "bb637f3ba152005be77e67ac5a5e1020")
(setf (get sensor::SensorImages :datatype-) "sensor/SensorImages")
(setf (get sensor::SensorImages :definition-)
      "sensor_msgs/Image input
sensor_msgs/Image motion
================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of cameara
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
# 0: no frame
# 1: global frame
string frame_id

")



(provide :sensor/SensorImages "bb637f3ba152005be77e67ac5a5e1020")


