#!/usr/bin/env bash
set -e

# ===== root視点でROSを通す（失敗しても落ちないように）=====
source /opt/ros/humble/setup.bash || true

# X11（必要なら）
if [ -n "${DISPLAY:-}" ] && [ -S /tmp/.X11-unix/X0 ]; then
  xhost +local:root >/dev/null 2>&1 || true
fi

# ===== ホストと同じユーザーを作って切替 =====
if [ -n "${HOST_USER:-}" ] && [ -n "${USER_ID:-}" ] && [ -n "${GROUP_ID:-}" ]; then
  # グループ
  if ! getent group "${GROUP_ID}" >/dev/null 2>&1; then
    groupadd -g "${GROUP_ID}" "${HOST_USER}" || true
  fi
  # ユーザー
  if ! id -u "${HOST_USER}" >/dev/null 2>&1; then
    useradd -m -u "${USER_ID}" -g "${GROUP_ID}" -s /bin/bash "${HOST_USER}"
  fi

  USER_HOME="/home/${HOST_USER}"

  # ユーザーの .bashrc にROSを通す
  if ! grep -q "opt/ros/humble/setup.bash" "${USER_HOME}/.bashrc" 2>/dev/null; then
    echo 'source /opt/ros/humble/setup.bash' >> "${USER_HOME}/.bashrc"
  fi

  # 目的ディレクトリを必ず用意
  mkdir -p "${USER_HOME}/yolov8"
  chown -R "${USER_ID}:${GROUP_ID}" "${USER_HOME}"

  # 開始ディレクトリは compose 側でも指定するが、念のためここでも移動
  cd "${USER_HOME}/yolov8" || true

  echo "[INFO] Switching to user ${HOST_USER} (uid:${USER_ID} gid:${GROUP_ID})"
  exec gosu "${HOST_USER}" "$@"
fi

# フォールバック（rootのまま）
exec "$@"