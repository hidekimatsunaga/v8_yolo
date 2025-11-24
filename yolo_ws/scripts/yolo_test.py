from ultralytics import YOLO
import cv2

# 1. 学習済みのYOLOv8モデルをロード
model = YOLO('model/0828.pt')

# 2. Webカメラに接続
# 0は通常、内蔵されているメインカメラを指します。
# USBカメラなどを接続している場合は1, 2などに変更してみてください。
cap = cv2.VideoCapture(5)

# カメラが正しく開かれたか確認
if not cap.isOpened():
    print("エラー: カメラを開けませんでした。")
    exit()

# 3. カメラからの映像をリアルタイムで処理
while True:
    # カメラから1フレーム（1枚の画像）を読み込む
    ret, frame = cap.read()
    if not ret:
        print("エラー: フレームを読み込めませんでした。")
        break

    # 読み込んだフレームに対して物体検出を実行
    results = model(frame)

    # 検出結果をフレームに描画
    annotated_frame = results[0].plot()

    # 結果が描画されたフレームを画面に表示
    cv2.imshow("Real-time YOLOv8 Detection", annotated_frame)

    # 'q'キーが押されたらループを終了してプログラムを閉じる
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. 使用したリソースを解放
cap.release()
cv2.destroyAllWindows()