import cv2
import os
import glob

# 切り出し画像が入っているフォルダ
input_dir = "/home/matsunaga-h/yolo_ws/grabcut_result"  # 必要ならパスを変更（例："./cutouts/"）

# 出力ファイルプレフィックス（例：result_output_001.png → mask_output_001.png）
input_prefix = "image"
output_prefix = "mask_output"

# PNG画像すべてを対象に処理
image_paths = glob.glob(os.path.join(input_dir, f"{input_prefix}_*.jpg"))

if not image_paths:
    print("処理対象の画像が見つかりませんでした。ファイル名やパスを確認してください。")
    exit()

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"読み込み失敗: {path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    filename = os.path.basename(path)
    out_name = filename.replace(input_prefix, output_prefix)
    out_path = os.path.join(input_dir, out_name)

    cv2.imwrite(out_path, mask)
    print(f"✔ 作成: {out_path}")

print("✅ すべてのマスク画像を生成しました。")
