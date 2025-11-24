# import cv2
# from ultralytics import YOLO

# # モデルのロード（yolov8n.pt, yolov8s.pt, yolov8m.ptなど）
# model = YOLO("/home/matsunaga-h/yolo_ws/model/jisaku.pt")  # 軽量なモデルを例示

# # カメラ起動（0はデフォルトカメラ）
# cap = cv2.VideoCapture(4)
# # cap = cv2.VideoCapture('/root/src/lena.jpg')


# if not cap.isOpened():
#     print("カメラが開けません")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("フレームの読み込みに失敗しました")
#         break

#     # YOLOで推論
#     results = model(frame)

#     # 推論結果の描画
#     annotated_frame = results[0].plot()

#     # 表示
#     cv2.imshow("YOLOv8 Detection", annotated_frame)

#     # 'q'で終了
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# #
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# モデルのロード（yolov8n.pt, yolov8s.pt, yolov8m.ptなど）
model = YOLO("/home/matsunaga-h/yolo_ws/model/jisaku.pt")  # ご自身のモデルパスを指定してください

# RealSenseパイプラインの初期化
pipeline = rs.pipeline()
config = rs.config()

# カラーストリームの設定 (解像度、フォーマット、フレームレート)
# お使いのカメラに合わせて解像度やフレームレートを調整してください
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# ストリーミング開始
pipeline.start(config)

print("カメラの起動が完了しました。'q'キーで終了します。")

try:
    while True:
        # フレームの待機
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            print("カラーフレームが取得できませんでした")
            continue

        # RealSenseのフレームをOpenCVで扱えるNumpy配列に変換
        frame = np.asanyarray(color_frame.get_data())

        # YOLOで推論
        results = model(frame)

        # 推論結果の描画
        annotated_frame = results[0].plot()

        # 表示
        cv2.imshow("YOLOv8 Detection with RealSense", annotated_frame)

        # 'q'で終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # ストリーミング停止
    print("カメラを停止します。")
    pipeline.stop()
    cv2.destroyAllWindows()