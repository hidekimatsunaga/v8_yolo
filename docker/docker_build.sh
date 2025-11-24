#!/usr/bin/env bash
set -euo pipefail

IMAGE=custom-ultralytics:humble-cpu

# スクリプトのあるディレクトリに移動
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "[*] Build ${IMAGE}"
DOCKER_BUILDKIT=1 docker build -t "${IMAGE}" -f Dockerfile .
echo "[*] Done."