import os

def count_each_digit(directory):
    # 0~6のカウント用辞書を初期化
    counts = {str(i): 0 for i in range(7)}

    # ディレクトリ内のすべてのファイルを探索
    for filename in os.listdir(directory):
        # 拡張子がtxtのファイルに限定
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    for line in file:
                        line = line.strip()
                        if line and line[0] in counts:
                            counts[line[0]] += 1
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    return counts

# 使用例
directory_path = '/root/yolo_ws/learning_data/train/labels'  # 対象のディレクトリパスに変更
result = count_each_digit(directory_path)

# 結果を表示
for digit in range(7):
    print(f"Lines starting with {digit}: {result[str(digit)]}")
