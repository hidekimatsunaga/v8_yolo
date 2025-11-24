#!/bin/bash
set -e

# ROS 2 環境のセットアップ
source /opt/ros/humble/setup.bash
# 環境変数などを使ってホームディレクトリを指定する
source $HOME/yolo_ws/install/setup.bash
# または、ボリュームマウントした先のパスを直接指定する
source /home/matsunaga-h/yolo_ws/install/setup.bash
# このスクリプトに渡されたコマンドを実行する
# (例: compose.yaml の command: bash など)
exec "$@"