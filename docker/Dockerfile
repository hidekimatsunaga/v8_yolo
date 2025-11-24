# ROS 2 Humble (Ubuntu 22.04) / CPU版
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    LANG=C.UTF-8 \
    # OpenCV のバックエンド優先度（GStreamer切り、V4L2最優先は任意）
    OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0 \
    OPENCV_VIDEOIO_PRIORITY_V4L2=1000

# 基本ツール
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release locales tzdata gosu pkg-config \
 && rm -rf /var/lib/apt/lists/*

# ロケール
RUN locale-gen en_US.UTF-8 ja_JP.UTF-8 && update-locale LANG=ja_JP.UTF-8

# ===== OpenCV を apt 版に統一（GUIあり）＋ V4L2 / GStreamer 依存 =====
# まず pip 版の OpenCV を完全排除（残骸も含めて削除）
RUN apt-get update && apt-get install -y python3-pip && \
    python3 -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true && \
    rm -rf /usr/local/lib/python3.*/dist-packages/cv2* /usr/local/lib/python3.*/site-packages/cv2* || true

# GUI & Video 入力関連
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-opencv \
    libgtk-3-0 libgtk-3-dev \
    v4l-utils \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
 && rm -rf /var/lib/apt/lists/*

# ===== ROS 2 Humble (Ubuntu 22.04 jammy) =====
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-ros-base \
      ros-humble-rclpy \
      ros-humble-image-transport \
      ros-humble-cv-bridge \
      ros-humble-vision-msgs \
      python3-colcon-common-extensions \
      ros-humble-rmw-cyclonedds-cpp \
 && rm -rf /var/lib/apt/lists/*

# ===== Ultralytics（依存は個別導入で opencv を絶対入れない） =====
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
        # Ultralytics 本体だけ入れて、依存は個別に（opencv 系は入れない）
        python3 -m pip install --no-cache-dir 'ultralytics==8.*' --no-deps && \
        # 必要最低限の依存（一部パッケージは PyPI、PyTorch は公式 CPU wheel を使う）
        python3 -m pip install --no-cache-dir \
            "numpy<2" \
            requests polars ultralytics-thop \
            pillow \
            pyyaml \
            tqdm \
            scipy \
            matplotlib \
            pandas \
            seaborn \
            psutil \
            onnx \
            onnxruntime && \
        # PyTorch (CPU wheels) を明示的に公式ホイールからインストール
        python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
        # 念のため再度 pip 版 OpenCV を潰し、apt 版を保証
        python3 -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true && \
        rm -rf /usr/local/lib/python3.*/dist-packages/cv2* /usr/local/lib/python3.*/site-packages/cv2* || true && \
        apt-get update && apt-get install -y --no-install-recommends python3-opencv && \
        rm -rf /var/lib/apt/lists/* && \
        # ビルド時にどの cv2 を読むかを確認（ここで失敗したらすぐ気付ける）
        python3 -c "import cv2; print('USING_CV2=', cv2.__file__, 'VER=', cv2.__version__); assert '/usr/lib/python3' in cv2.__file__, 'pip版のOpenCVが残っている可能性があります'"

# 作業ディレクトリ
WORKDIR /workspace

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# エントリポイント
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]