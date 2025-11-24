# import os

# # 処理対象ディレクトリ
# target_dir = "/home/matsunaga-h/yolo_ws/jissainogomi"
# ext = ".jpg"

# # すでにリネーム済みのファイルを取得（image_XXX.jpg 形式）
# renamed_files = sorted([f for f in os.listdir(target_dir) if f.startswith("image_") and f.endswith(ext)])

# # 推定で元のファイル名に戻す
# for i, filename in enumerate(renamed_files, start=10):  # 例として10から開始
#     new_name = f"Image ({i}).jpg"
#     src = os.path.join(target_dir, filename)
#     dst = os.path.join(target_dir, new_name)
#     os.rename(src, dst)
#     print(f"{filename} → {new_name}")
import os
import re

# 処理対象のディレクトリ
target_dir = "/home/matsunaga-h/yolo_ws/src/grabcut_result"
ext = ".jpg"

# ディレクトリ内のファイル取得
jpg_files = [f for f in os.listdir(target_dir) if f.lower().endswith(ext)]

# 正規表現でファイル名から数字を抽出し、タプルにしてソート
file_infos = []
for filename in jpg_files:
    match = re.search(r"(\d+)", filename)
    if match:
        number = int(match.group(1))
        file_infos.append((number, filename))
    else:
        print(f"⚠ 数字が見つからないファイル: {filename}")

# 数字順に並べ替え
file_infos.sort()

# リネーム実行
for number, filename in file_infos:
    new_name = f"image_{number:03}{ext}"
    src = os.path.join(target_dir, filename)
    dst = os.path.join(target_dir, new_name)
    os.rename(src, dst)
    print(f"{filename} → {new_name}")
