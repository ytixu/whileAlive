
(cl:in-package :asdf)

(defsystem "sensor-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :sensor_msgs-msg
)
  :components ((:file "_package")
    (:file "SensorImages" :depends-on ("_package_SensorImages"))
    (:file "_package_SensorImages" :depends-on ("_package"))
  ))