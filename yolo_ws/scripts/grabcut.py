import cv2
import numpy as np

# 画像読み込み
# img_path = "/home/matsunaga-h/yolo_ws/jissainogomi/Image (7).jpg"  # ファイル名が合っていることを確認
img_path = "Image (7).jpg"  # ファイル名が合っていることを確認
img = cv2.imread(img_path)

# 読み込めたか確認
if img is None:
    raise FileNotFoundError(f"画像の読み込みに失敗しました: {img_path}")

# ユーザーがROIをマウスで選択
rect = cv2.selectROI("画像をドラッグして対象を囲んでください", img, False, False)
cv2.destroyAllWindows()  # 選択後ウィンドウ閉じる

mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# GrabCut実行
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
result = img * mask2[:, :, np.newaxis]

# 結果表示・保存
cv2.imshow("マスク画像", mask2 * 255)
cv2.imshow("切り出し結果", result)
cv2.imwrite("mask_output.png", mask2 * 255)
cv2.imwrite("result_output.png", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
