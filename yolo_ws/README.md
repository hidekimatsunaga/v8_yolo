# yolo_ws

## dockerの入り方
- ./docker_yolo.sh

## dockerのbuild方法
- docker image build -t yolov8_ros:latest .

## yolo学習データ水増し方法
### 名前を変更
- python3 scripts/name.py
### grabcut 対象物が写った画像を前景と背景に分ける
- python3 scripts/batch_grabcut.py
### mask画像を生成
- python3 scripts/mask.py
#### 対象物と背景画像を合成してyoloの学習データを作成
- python3 scripts/generate_synthetic_data.py 
