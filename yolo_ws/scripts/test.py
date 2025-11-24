import argparse
import glob
import os

import cv2


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str)
    parser.add_argument("--save_dir", type=str, default="./litter")
    return parser.parse_args()



def crop_bbox(image_path: str, save_dir: str) -> None:
    img = cv2.imread(image_path)
    name = os.path.splitext(os.path.basename(image_path))[0]

    # スケーリング係数を計算（長辺を最大1000pxに）
    max_length = 1000
    height, width = img.shape[:2]
    scale = min(1.0, max_length / max(height, width))  # 1未満なら縮小

    if scale < 1.0:
        resized_img = cv2.resize(img, (int(width * scale), int(height * scale)))
    else:
        resized_img = img.copy()

    cnt = len(glob.glob(os.path.join(save_dir, f"{name}*.jpg")))

    while True:
        ROI = cv2.selectROI(
            "Please draw a bounding box", resized_img, fromCenter=False, showCrosshair=False
        )
        left, top, w, h = map(int, ROI)

        if w == 0 or h == 0:
            break

        # 縮小画像でのROIを元画像サイズに換算
        left_orig = int(left / scale)
        top_orig = int(top / scale)
        w_orig = int(w / scale)
        h_orig = int(h / scale)

        cropped_img = img[top_orig : top_orig + h_orig, left_orig : left_orig + w_orig]
        cv2.imshow("crop", cropped_img)

        while True:
            k = cv2.waitKey(0)
            if k == ord("r"):
                cv2.destroyWindow("crop")
                break
            elif k == ord("s"):
                cnt += 1
                save_path = os.path.join(save_dir, f"{name}{cnt:0>3}.jpg")
                cv2.imwrite(save_path, cropped_img)
                cv2.destroyWindow("crop")
                break


def main():
    args = get_arguments()

    os.makedirs(args.save_dir, exist_ok=True)

    if not os.path.exists(args.img_path):
        raise FileNotFoundError
    else:
        crop_bbox(args.img_path, args.save_dir)


if __name__ == "__main__":
    main()