import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
import sys

# RealSenseデバイスの接続チェック
ctx = rs.context()
if len(ctx.devices) == 0:
    print("❌ RealSenseカメラが接続されていません。USBケーブルを確認してください。")
    sys.exit(1)
else:
    print("✅ RealSenseカメラが接続されました。")


# YOLOモデルの読み込み
model = YOLO("model/ultimate_cigarette.pt")

# RealSense設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 距離スケール取得
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align = rs.align(rs.stream.color)

try:
    while True:
        # フレーム取得とアライメント
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # YOLO推論
        results = model(color_image)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                depth_value = depth_image[cy, cx] * depth_scale
                print(f"物体中心 ({cx}, {cy}) の深度: {depth_value:.2f} m")

                # 可視化
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_image, f"{depth_value:.2f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("YOLO with Depth", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
