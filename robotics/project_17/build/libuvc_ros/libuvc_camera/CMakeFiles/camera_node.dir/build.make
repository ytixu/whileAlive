# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ytixu/gitHTML/whileAlive/robotics/project_17/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ytixu/gitHTML/whileAlive/robotics/project_17/build

# Include any dependencies generated for this target.
include libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/depend.make

# Include the progress variables for this target.
include libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/progress.make

# Include the compile flags for this target's objects.
include libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/flags.make

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.o: libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/flags.make
libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.o: /home/ytixu/gitHTML/whileAlive/robotics/project_17/src/libuvc_ros/libuvc_camera/src/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ytixu/gitHTML/whileAlive/robotics/project_17/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.o"
	cd /home/ytixu/gitHTML/whileAlive/robotics/project_17/build/libuvc_ros/libuvc_camera && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/camera_node.dir/src/main.cpp.o -c /home/ytixu/gitHTML/whileAlive/robotics/project_17/src/libuvc_ros/libuvc_camera/src/main.cpp

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/camera_node.dir/src/main.cpp.i"
	cd /home/ytixu/gitHTML/whileAlive/robotics/project_17/build/libuvc_ros/libuvc_camera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ytixu/gitHTML/whileAlive/robotics/project_17/src/libuvc_ros/libuvc_camera/src/main.cpp > CMakeFiles/camera_node.dir/src/main.cpp.i

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/camera_node.dir/src/main.cpp.s"
	cd /home/ytixu/gitHTML/whileAlive/robotics/project_17/build/libuvc_ros/libuvc_camera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ytixu/gitHTML/whileAlive/robotics/project_17/src/libuvc_ros/libuvc_camera/src/main.cpp -o CMakeFiles/camera_node.dir/src/main.cpp.s

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.o.requires:
.PHONY : libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.o.requires

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.o.provides: libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.o.requires
	$(MAKE) -f libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/build.make libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.o.provides.build
.PHONY : libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.o.provides

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.o.provides.build: libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.o

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.o: libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/flags.make
libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.o: /home/ytixu/gitHTML/whileAlive/robotics/project_17/src/libuvc_ros/libuvc_camera/src/camera_driver.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ytixu/gitHTML/whileAlive/robotics/project_17/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.o"
	cd /home/ytixu/gitHTML/whileAlive/robotics/project_17/build/libuvc_ros/libuvc_camera && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/camera_node.dir/src/camera_driver.cpp.o -c /home/ytixu/gitHTML/whileAlive/robotics/project_17/src/libuvc_ros/libuvc_camera/src/camera_driver.cpp

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/camera_node.dir/src/camera_driver.cpp.i"
	cd /home/ytixu/gitHTML/whileAlive/robotics/project_17/build/libuvc_ros/libuvc_camera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ytixu/gitHTML/whileAlive/robotics/project_17/src/libuvc_ros/libuvc_camera/src/camera_driver.cpp > CMakeFiles/camera_node.dir/src/camera_driver.cpp.i

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/camera_node.dir/src/camera_driver.cpp.s"
	cd /home/ytixu/gitHTML/whileAlive/robotics/project_17/build/libuvc_ros/libuvc_camera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ytixu/gitHTML/whileAlive/robotics/project_17/src/libuvc_ros/libuvc_camera/src/camera_driver.cpp -o CMakeFiles/camera_node.dir/src/camera_driver.cpp.s

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.o.requires:
.PHONY : libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.o.requires

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.o.provides: libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.o.requires
	$(MAKE) -f libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/build.make libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.o.provides.build
.PHONY : libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.o.provides

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.o.provides.build: libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.o

# Object files for target camera_node
camera_node_OBJECTS = \
"CMakeFiles/camera_node.dir/src/main.cpp.o" \
"CMakeFiles/camera_node.dir/src/camera_driver.cpp.o"

# External object files for target camera_node
camera_node_EXTERNAL_OBJECTS =

/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.o
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.o
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/build.make
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libuvc.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libcamera_info_manager.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libdynamic_reconfigure_config_init_mutex.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libimage_transport.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libmessage_filters.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libnodeletlib.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libbondcpp.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libuuid.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libtinyxml.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libclass_loader.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/libPocoFoundation.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libdl.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libroslib.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libroscpp.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libboost_signals.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/librosconsole.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/librosconsole_log4cxx.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/librosconsole_backend_interface.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/liblog4cxx.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libboost_regex.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libxmlrpcpp.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libroscpp_serialization.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/librostime.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libcpp_common.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libcamera_info_manager.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libdynamic_reconfigure_config_init_mutex.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libimage_transport.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libmessage_filters.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libnodeletlib.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libbondcpp.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libuuid.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libtinyxml.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libclass_loader.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/libPocoFoundation.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libdl.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libroslib.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libroscpp.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libboost_signals.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/librosconsole.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/librosconsole_log4cxx.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/librosconsole_backend_interface.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/liblog4cxx.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libboost_regex.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libxmlrpcpp.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libroscpp_serialization.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/librostime.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /opt/ros/indigo/lib/libcpp_common.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
/home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node: libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable /home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node"
	cd /home/ytixu/gitHTML/whileAlive/robotics/project_17/build/libuvc_ros/libuvc_camera && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/camera_node.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/build: /home/ytixu/gitHTML/whileAlive/robotics/project_17/devel/lib/libuvc_camera/camera_node
.PHONY : libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/build

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/requires: libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/main.cpp.o.requires
libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/requires: libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/src/camera_driver.cpp.o.requires
.PHONY : libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/requires

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/clean:
	cd /home/ytixu/gitHTML/whileAlive/robotics/project_17/build/libuvc_ros/libuvc_camera && $(CMAKE_COMMAND) -P CMakeFiles/camera_node.dir/cmake_clean.cmake
.PHONY : libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/clean

libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/depend:
	cd /home/ytixu/gitHTML/whileAlive/robotics/project_17/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ytixu/gitHTML/whileAlive/robotics/project_17/src /home/ytixu/gitHTML/whileAlive/robotics/project_17/src/libuvc_ros/libuvc_camera /home/ytixu/gitHTML/whileAlive/robotics/project_17/build /home/ytixu/gitHTML/whileAlive/robotics/project_17/build/libuvc_ros/libuvc_camera /home/ytixu/gitHTML/whileAlive/robotics/project_17/build/libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libuvc_ros/libuvc_camera/CMakeFiles/camera_node.dir/depend

