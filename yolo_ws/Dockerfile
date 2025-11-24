# ベースイメージ: ROS2 Humble
FROM ros:humble

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/opt/librealsense/build/wrappers/python:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# -----------------------------------------------------------------------------
# APTパッケージのセットアップ
# -----------------------------------------------------------------------------

# ROS 2 の APTリポジトリを再設定
RUN rm -f /etc/apt/sources.list.d/ros2-latest.list && \
    apt-get update -q && \
    apt-get install -yq --no-install-recommends curl && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update -q && \
    rm -rf /var/lib/apt/lists/*

# システムとROSの依存パッケージをインストール
RUN apt-get update -q && apt-get install -yq --no-install-recommends \
    # Python関連
    python3-pip \
    python3-colcon-common-extensions \
    python3-colcon-ros \
    # GUIおよび画像処理関連
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    # RealSenseビルド用
    libusb-1.0-0-dev \
    libgtk-3-dev \
    libglfw3-dev \
    libglu1-mesa-dev \
    # ビルドツール
    cmake \
    git \
    build-essential \
    # ROSパッケージ
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Pythonライブラリのセットアップ
# -----------------------------------------------------------------------------

# pipで管理する主要ライブラリのバージョンを整理・再インストール
# APTでインストールされたバージョンと競合を避けるため
RUN pip uninstall -y opencv-python opencv-python-headless matplotlib numpy && \
    rm -rf /usr/local/lib/python3.10/dist-packages/matplotlib* && \
    pip install --no-cache-dir \
    numpy==1.24.4 \
    opencv-python \
    matplotlib==3.7.1

# requirements.txt からPythonライブラリをインストール
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir --upgrade setuptools packaging wheel

# -----------------------------------------------------------------------------
# Intel RealSense SDK (librealsense) のビルドとインストール
# -----------------------------------------------------------------------------
WORKDIR /opt
RUN git clone https://github.com/IntelRealSense/librealsense.git -b v2.54.1 && \
    cd librealsense && \
    mkdir -p build && cd build && \
    cmake .. -DBUILD_PYTHON_BINDINGS=ON \
             -DPYTHON_EXECUTABLE=$(which python3) \
             -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_EXAMPLES=OFF \
             -DBUILD_GRAPHICAL_EXAMPLES=OFF && \
    make -j$(nproc) && \
    make install

# -----------------------------------------------------------------------------
# ROS 2 ワークスペースのセットアップ
# -----------------------------------------------------------------------------
# ワークスペースディレクトリを作成し、ソースコードをコピー
WORKDIR /root/yolo_ws
COPY . .
# pipとsetuptoolsをアップグレード
RUN pip install --upgrade pip setuptools
# 依存関係を解決し、指定パッケージをビルド
RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select yolov8_ros

# -----------------------------------------------------------------------------
# 環境設定と最終処理
# -----------------------------------------------------------------------------

# .bashrc に環境設定を追記し、コンテナ起動時に自動で読み込むようにする
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /root/yolo_ws/install/setup.bash" >> ~/.bashrc && \
    echo "export PYTHONPATH=/opt/librealsense/build/wrappers/python:\$PYTHONPATH" >> ~/.bashrc

# 不要なキャッシュを削除
RUN rm -rf ~/.cache/pip
# 作業ディレクトリをホームに設定
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]

CMD ["bash"]