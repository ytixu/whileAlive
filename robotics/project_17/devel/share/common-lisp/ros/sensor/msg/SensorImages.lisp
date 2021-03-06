; Auto-generated. Do not edit!


(cl:in-package sensor-msg)


;//! \htmlinclude SensorImages.msg.html

(cl:defclass <SensorImages> (roslisp-msg-protocol:ros-message)
  ((input
    :reader input
    :initarg :input
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image))
   (motion
    :reader motion
    :initarg :motion
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image))
   (segment_viz
    :reader segment_viz
    :initarg :segment_viz
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image)))
)

(cl:defclass SensorImages (<SensorImages>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SensorImages>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SensorImages)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name sensor-msg:<SensorImages> is deprecated: use sensor-msg:SensorImages instead.")))

(cl:ensure-generic-function 'input-val :lambda-list '(m))
(cl:defmethod input-val ((m <SensorImages>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sensor-msg:input-val is deprecated.  Use sensor-msg:input instead.")
  (input m))

(cl:ensure-generic-function 'motion-val :lambda-list '(m))
(cl:defmethod motion-val ((m <SensorImages>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sensor-msg:motion-val is deprecated.  Use sensor-msg:motion instead.")
  (motion m))

(cl:ensure-generic-function 'segment_viz-val :lambda-list '(m))
(cl:defmethod segment_viz-val ((m <SensorImages>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sensor-msg:segment_viz-val is deprecated.  Use sensor-msg:segment_viz instead.")
  (segment_viz m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SensorImages>) ostream)
  "Serializes a message object of type '<SensorImages>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'input) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'motion) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'segment_viz) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SensorImages>) istream)
  "Deserializes a message object of type '<SensorImages>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'input) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'motion) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'segment_viz) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SensorImages>)))
  "Returns string type for a message object of type '<SensorImages>"
  "sensor/SensorImages")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SensorImages)))
  "Returns string type for a message object of type 'SensorImages"
  "sensor/SensorImages")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SensorImages>)))
  "Returns md5sum for a message object of type '<SensorImages>"
  "915977008f098b0d3dd8dab35bc4244c")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SensorImages)))
  "Returns md5sum for a message object of type 'SensorImages"
  "915977008f098b0d3dd8dab35bc4244c")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SensorImages>)))
  "Returns full string definition for message of type '<SensorImages>"
  (cl:format cl:nil "sensor_msgs/Image input~%sensor_msgs/Image motion~%sensor_msgs/Image segment_viz~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of cameara~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SensorImages)))
  "Returns full string definition for message of type 'SensorImages"
  (cl:format cl:nil "sensor_msgs/Image input~%sensor_msgs/Image motion~%sensor_msgs/Image segment_viz~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of cameara~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SensorImages>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'input))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'motion))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'segment_viz))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SensorImages>))
  "Converts a ROS message object to a list"
  (cl:list 'SensorImages
    (cl:cons ':input (input msg))
    (cl:cons ':motion (motion msg))
    (cl:cons ':segment_viz (segment_viz msg))
))
