# yolo_ws

ROS 2 (Humble) 上で YOLOv8 推論と深度情報の取得を行うワークスペース。
RealSense の RGB/Depth トピックを受け取り、検出結果の 3D 座標と BB 情報を publish します。

## 構成
- `src/yolov8_ros/` : ROS 2 ノード (`yolo_3d_node`) の本体
- `model/` : 学習済みモデル (`*.pt`)
- `learning_test/` : 学習/検証データ
- `scripts/` : データ作成/補助スクリプト群
- `data.yaml` : YOLO 用データセット定義
- `yolov8n.pt` : ベースモデル

## Docker ビルド
```bash
docker image build -t yolov8_ros:latest .
```

## Docker 起動
### rocker で起動 (推奨)
```bash
./docker_yolo.sh
```

### 素の docker で起動
```bash
./docker_yolo_test.sh
```
初回は `xhost +local:docker` を実行して GUI を許可してください。
ホスト側のパスが異なる場合は `docker_yolo.sh` / `docker_yolo_test.sh` の
`/home/matsunaga-h/yolo_ws` を自分の環境に合わせて修正します。

## ノード実行
コンテナ内で:
```bash
ros2 run yolov8_ros yolo_3d_node
```

GUI ウィンドウ位置を固定したい場合は、以下のパラメータを指定できます。
```bash
ros2 run yolov8_ros yolo_3d_node --ros-args -p screen_width:=1920 -p screen_height:=1080
```

## キー操作 (モデル切り替え)
- `n` : 次のモデル
- `p` : 前のモデル
- `1..9` : 直接選択

モデルのパスは [src/yolov8_ros/yolov8_ros/yolo_3d_node.py](src/yolov8_ros/yolov8_ros/yolo_3d_node.py) の
`model_paths` を編集してください。

## トピック
### Subscribe
- `/camera/camera/color/image_raw` (sensor_msgs/Image)
- `/camera/camera/aligned_depth_to_color/image_raw` (sensor_msgs/Image)
- `/camera/camera/aligned_depth_to_color/camera_info` (sensor_msgs/CameraInfo)

### Publish
- `/detected_depth_points` (geometry_msgs/PointStamped)
- `/yolo/bb_locked` (std_msgs/Float32MultiArray)
  - `data = [object_id, class_id, conf, cx, cy, bw, bh, area]`

## データ水増しフロー
1. 画像/ラベルの名前を整える
	```bash
	python3 scripts/name.py
	```
2. grabcut で前景/背景分離
	```bash
	python3 scripts/batch_grabcut.py
	```
3. マスク生成
	```bash
	python3 scripts/mask.py
	```
4. 前景と背景を合成して学習データ作成
	```bash
	python3 scripts/generate_synthetic_data.py
	```

## 学習 (参考)
Ultralytics CLI を使う場合の例:
```bash
yolo train model=yolov8n.pt data=data.yaml
```

## 便利スクリプト
- [scripts/class_count.py](scripts/class_count.py) : クラス数の集計
- [scripts/class_id_converter.py](scripts/class_id_converter.py) : クラスIDの変換
- [scripts/heic_to_jpg.py](scripts/heic_to_jpg.py) : HEIC 変換

