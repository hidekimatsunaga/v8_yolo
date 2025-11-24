import cv2
import os
import glob
import numpy as np
import random

# 入力パス
background_dir = "/home/matsunaga-h/yolo_ws/haikei"
garbage_dir = "/home/matsunaga-h/yolo_ws/grabcut_result"

# 出力パス
output_image_dir = "/home/matsunaga-h/yolo_ws/cigarette/images"
output_label_dir = "/home/matsunaga-h/yolo_ws/cigarette/labels"
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# ゴミ画像とマスク画像を対応させて取得
garbage_imgs = sorted(glob.glob(os.path.join(garbage_dir, "image_*.jpg")))
mask_imgs = sorted(glob.glob(os.path.join(garbage_dir, "mask_output_*.jpg")))
background_imgs = glob.glob(os.path.join(background_dir, "*.jpg"))

# YOLO用 正規化関数
def convert_bbox(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return x_center, y_center, w, h

# 合成枚数
# num_images_to_generate = min(len(background_imgs), 100)
num_images_to_generate = 300

for i in range(num_images_to_generate):
    bg_path = random.choice(background_imgs)
    bg_img = cv2.imread(bg_path)
    bg_h, bg_w = bg_img.shape[:2]

    # ランダムにゴミ画像とマスクを選択
    idx = random.randint(0, len(garbage_imgs) - 1)
    fg_path = garbage_imgs[idx]
    mask_path = mask_imgs[idx]
    fg_img = cv2.imread(fg_path)
    mask = cv2.imread(mask_path, 0)  # グレースケール

    # リサイズ（ランダムなスケール）
    scale = random.uniform(0.1, 0.3)
    new_w = int(fg_img.shape[1] * scale)
    new_h = int(fg_img.shape[0] * scale)
    fg_img = cv2.resize(fg_img, (new_w, new_h))
    mask = cv2.resize(mask, (new_w, new_h))

    # ランダム位置
    max_x = bg_w - new_w
    max_y = bg_h - new_h
    if max_x <= 0 or max_y <= 0:
        print(f"⚠ スキップ: ゴミ画像が背景より大きすぎます ({fg_path})")
        continue
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # 合成処理
    roi = bg_img[y:y+new_h, x:x+new_w]
    roi[mask == 255] = fg_img[mask == 255]
    bg_img[y:y+new_h, x:x+new_w] = roi

    # 保存
    output_img_name = f"synthetic_{i:03}.jpg"
    output_img_path = os.path.join(output_image_dir, output_img_name)
    cv2.imwrite(output_img_path, bg_img)

    # YOLO形式ラベル作成（1クラス=0）
    x_center, y_center, w_norm, h_norm = convert_bbox(x, y, new_w, new_h, bg_w, bg_h)
    label_path = os.path.join(output_label_dir, output_img_name.replace(".jpg", ".txt"))
    with open(label_path, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    print(f"✅ 作成: {output_img_path} ＋ アノテーション")

print("🎉 合成画像とラベルの生成が完了しました！")
