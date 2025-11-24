import cv2
import os
import glob
import numpy as np

# 画像フォルダ
image_folder = os.path.join(os.path.dirname(__file__), "../litters/cigarette")
save_folder = os.path.join(os.path.dirname(__file__), "../grabcut_result")
os.makedirs(save_folder, exist_ok=True)

# 画像一覧取得
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
print(f"処理対象画像数: {len(image_paths)}")

for path in image_paths:
    print(f"処理中: {path}")
    img = cv2.imread(path)
    if img is None:
        print(f"読み込み失敗: {path}")
        continue

    # 画像リサイズ（幅640に調整、高さはアスペクト比保持）
    h, w = img.shape[:2]
    new_width = 640
    scale = new_width / w
    resized_img = cv2.resize(img, (new_width, int(h * scale)))

    # ROI選択（リサイズ画像で）
    roi = cv2.selectROI("ROIEnter", resized_img, fromCenter=False, showCrosshair=True)
    if roi == (0, 0, 0, 0):
        print("ROI未選択、スキップ")
        continue

    # ROIを元画像サイズにスケーリングし直す
    x, y, rw, rh = roi
    x = int(x / scale)
    y = int(y / scale)
    rw = int(rw / scale)
    rh = int(rh / scale)
    fullsize_roi = (x, y, rw, rh)

    # GrabCut実行（元画像に対して）
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, fullsize_roi, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    result = img * mask2[:, :, np.newaxis]

    filename = os.path.basename(path)
    save_path = os.path.join(save_folder, filename)
    cv2.imwrite(save_path, result)
    print(f"保存完了: {save_path}")

    cv2.imshow("結果", cv2.resize(result, (new_width, int(h * scale))))
    cv2.waitKey(0)

cv2.destroyAllWindows()

