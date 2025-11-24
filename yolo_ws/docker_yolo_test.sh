xhost +local:docker  # GUI許可（初回のみ）

docker run -it \
    --rm \
    --name yolov8_test \
    -v /dev:/dev \
    -v /dev/bus/usb:/dev/bus/usb \
    --privileged \
    --net=host \
    --env DISPLAY=$DISPLAY \
    --env QT_X11_NO_MITSHM=1 \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume /home/matsunaga-h/yolo_ws:/root/yolo_ws \
    yolov8_ros:latest