#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
    DESTDIR_ARG="--root=$DESTDIR"
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/ytixu/gitHTML/whileAlive/robotics/project_17/src/sensor"

# snsure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/ytixu/gitHTML/whileAlive/robotics/project_17/install/lib/python2.7/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/ytixu/gitHTML/whileAlive/robotics/project_17/install/lib/python2.7/dist-packages:/home/ytixu/gitHTML/whileAlive/robotics/project_17/build/lib/python2.7/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/ytixu/gitHTML/whileAlive/robotics/project_17/build" \
    "/usr/bin/python" \
    "/home/ytixu/gitHTML/whileAlive/robotics/project_17/src/sensor/setup.py" \
    build --build-base "/home/ytixu/gitHTML/whileAlive/robotics/project_17/build/sensor" \
    install \
    $DESTDIR_ARG \
    --install-layout=deb --prefix="/home/ytixu/gitHTML/whileAlive/robotics/project_17/install" --install-scripts="/home/ytixu/gitHTML/whileAlive/robotics/project_17/install/bin"
