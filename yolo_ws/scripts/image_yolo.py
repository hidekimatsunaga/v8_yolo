import cv2
from ultralytics import YOLO
import os
from glob import glob


model = YOLO("/root/yolo_ws/model/ultimate_cigarette.pt")

# 画像フォルダのパス（必要に応じて変更）
image_dir = "./jissainogomi"
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)

# 対象画像のパスを取得（jpg, pngなど対応）
image_paths = glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png"))

# 一括推論と保存
for img_path in image_paths:
    img = cv2.imread(img_path)
    results = model(img)
    annotated_img = results[0].plot()
    
    # 出力ファイル名を生成して保存
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, annotated_img)

print(f"処理完了: {len(image_paths)} 枚の画像に対して推論しました。")