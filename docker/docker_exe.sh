#!/bin/bash
set -e

CONTAINER_NAME=yolov8-cpu
SERVICE_NAME=yolov8

export HOST_USER=$(whoami)
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export HOST_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

cd "$(dirname "$0")"

xhost +SI:localuser:root

# コンテナを起動（バックグラウンド）
echo "[INFO] Starting container '${CONTAINER_NAME}' with docker compose up -d..."
docker compose up -d "${SERVICE_NAME}"

# コンテナが起動するまで待機
sleep 2

# execで中に入る
echo "[INFO] Entering container '${CONTAINER_NAME}'..."
docker exec -it "${CONTAINER_NAME}" bash

# bashを抜けたら自動でコンテナ停止＆削除
echo "[INFO] Container exited. Stopping and removing..."
docker compose down --volumes