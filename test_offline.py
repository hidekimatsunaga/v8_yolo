import cv2
import time

from ultralytics import YOLO

# model_path = "/home/ruby/yolov8/model/241206_signal_epoch225_batch32.pt"
# model_path = "/home/ruby/yolov8/model/241206_abc_epoch250_batch32.pt"
# model_path = "/home/ruby/yolov8/model/241206_bluebox_150_32.pt"
model_path = "/home/ruby/yolov8/model/bluebox_200_32.pt"
print(f"選択されたモデル: {model_path}")
model = YOLO(model_path)

# cap = cv2.VideoCapture("/home/ruby/yolov8/movie_2025/251005_selectB_clip.mp4")
# cap = cv2.VideoCapture("/home/ruby/yolov8/movie_2025/251005_selectC_clip.mp4")
cap = cv2.VideoCapture("/home/ruby/yolov8/movie_2025/251005_selectD1_bluebox_clip.mp4")
# cap = cv2.VideoCapture("/home/ruby/yolov8/movie_2025/251005_selectD1_label_clip.mp4")



# カメラ初期化（1920x1080
width, height = 1920, 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

if not cap.isOpened():
    print("Error: /dev/video0 を開けません")
    exit()

# 表示を縮小する倍率（例: 0.5で960x540表示）
scale_factor = 0.5
interval = 1.0 / 30  # 30FPS

print(f"YOLOv8リアルタイム推論開始 ({width}x{height})")

try:
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("フレーム読み取り失敗")
            break

        # YOLOv8推論（BGR→RGB）
        results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), verbose=False)
        annotated_frame = results[0].plot()

        # BGRに戻してから縮小して表示
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        resized_frame = cv2.resize(annotated_frame_bgr, (0, 0), fx=scale_factor, fy=scale_factor)
        cv2.imshow("YOLOv8 Detection (Resized)", resized_frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 5FPSになるようにスリープ
        elapsed = time.time() - start_time
        time.sleep(max(0, interval - elapsed))

except KeyboardInterrupt:
    print("中断されました")

finally:
    cap.release()
    cv2.destroyAllWindows()