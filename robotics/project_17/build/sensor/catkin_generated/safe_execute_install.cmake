execute_process(COMMAND "/home/ytixu/gitHTML/whileAlive/robotics/project_17/build/sensor/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/ytixu/gitHTML/whileAlive/robotics/project_17/build/sensor/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
