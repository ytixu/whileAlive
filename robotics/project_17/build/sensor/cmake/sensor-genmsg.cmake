# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "sensor: 1 messages, 0 services")

set(MSG_I_FLAGS "-Isensor:/home/ytixu/gitHTML/whileAlive/robotics/project_17/src/sensor/msg;-Isensor_msgs:/opt/ros/indigo/share/sensor_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/indigo/share/geometry_msgs/cmake/../msg;-Istd_msgs:/opt/ros/indigo/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(sensor_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/ytixu/gitHTML/whileAlive/robotics/project_17/src/sensor/msg/SensorImages.msg" NAME_WE)
add_custom_target(_sensor_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "sensor" "/home/ytixu/gitHTML/whileAlive/robotics/project_17/src/sensor/msg/SensorImages.msg" "std_msgs/Header:sensor_msgs/Image"
)

#
#  langs = gencpp;geneus;genlisp;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(sensor
  "/home/ytixu/gitHTML/whileAlive/robotics/project_17/src/sensor/msg/SensorImages.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/indigo/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/indigo/share/sensor_msgs/cmake/../msg/Image.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sensor
)

### Generating Services

### Generating Module File
_generate_module_cpp(sensor
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sensor
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(sensor_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(sensor_generate_messages sensor_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/ytixu/gitHTML/whileAlive/robotics/project_17/src/sensor/msg/SensorImages.msg" NAME_WE)
add_dependencies(sensor_generate_messages_cpp _sensor_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sensor_gencpp)
add_dependencies(sensor_gencpp sensor_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sensor_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(sensor
  "/home/ytixu/gitHTML/whileAlive/robotics/project_17/src/sensor/msg/SensorImages.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/indigo/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/indigo/share/sensor_msgs/cmake/../msg/Image.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sensor
)

### Generating Services

### Generating Module File
_generate_module_eus(sensor
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sensor
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(sensor_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(sensor_generate_messages sensor_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/ytixu/gitHTML/whileAlive/robotics/project_17/src/sensor/msg/SensorImages.msg" NAME_WE)
add_dependencies(sensor_generate_messages_eus _sensor_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sensor_geneus)
add_dependencies(sensor_geneus sensor_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sensor_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(sensor
  "/home/ytixu/gitHTML/whileAlive/robotics/project_17/src/sensor/msg/SensorImages.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/indigo/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/indigo/share/sensor_msgs/cmake/../msg/Image.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sensor
)

### Generating Services

### Generating Module File
_generate_module_lisp(sensor
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sensor
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(sensor_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(sensor_generate_messages sensor_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/ytixu/gitHTML/whileAlive/robotics/project_17/src/sensor/msg/SensorImages.msg" NAME_WE)
add_dependencies(sensor_generate_messages_lisp _sensor_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sensor_genlisp)
add_dependencies(sensor_genlisp sensor_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sensor_generate_messages_lisp)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(sensor
  "/home/ytixu/gitHTML/whileAlive/robotics/project_17/src/sensor/msg/SensorImages.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/indigo/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/indigo/share/sensor_msgs/cmake/../msg/Image.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sensor
)

### Generating Services

### Generating Module File
_generate_module_py(sensor
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sensor
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(sensor_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(sensor_generate_messages sensor_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/ytixu/gitHTML/whileAlive/robotics/project_17/src/sensor/msg/SensorImages.msg" NAME_WE)
add_dependencies(sensor_generate_messages_py _sensor_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sensor_genpy)
add_dependencies(sensor_genpy sensor_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sensor_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sensor)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sensor
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(sensor_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sensor)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sensor
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(sensor_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sensor)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sensor
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(sensor_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sensor)
  install(CODE "execute_process(COMMAND \"/usr/bin/python\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sensor\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sensor
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(sensor_generate_messages_py sensor_msgs_generate_messages_py)
endif()
