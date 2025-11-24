#!/bin/sh

rocker \
    --x11 \
    --user \
    --volume /home/matsunaga-h/yolo_ws:/home/matsunaga-h/yolo_ws \
    --volume /dev/shm:/dev/shm \
    --volume /dev:/dev \
    --name yolov8 \
    --network=host \
    -- yolov8_ros:latest