import os

# ラベルファイルのディレクトリを指定
label_dir = "/home/matsunaga-h/Downloads/Cigarette Bud Advanced.v1i.yolov8/valid/labels"

# クラスIDを置き換えるための関数
def replace_class_id(file_path, old_id="0", new_id="8"):
    with open(file_path, "r") as file:
        lines = file.readlines()
    # 0 を 2 に変更
    modified_lines = [line.replace(f"{old_id} ", f"{new_id} ", 1) if line.startswith(f"{old_id} ") else line for line in lines]
    with open(file_path, "w") as file:
        file.writelines(modified_lines)

# ディレクトリ内のすべてのラベルファイルを処理
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):  # 拡張子が.txtのファイルを対象
        file_path = os.path.join(label_dir, filename)
        replace_class_id(file_path)

print("クラスIDの置換が完了しました。")
