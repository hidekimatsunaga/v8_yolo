##HEICをjpgに変換するプログラム
import os
import subprocess   # subprocessを使って外部コマンドを実行  
import glob
import re
from PIL import Image
# 処理対象のディレクトリ
target_dir = "/home/matsunaga-h/litters/cigarette"
# HEICファイルの拡張子
heic_ext = ".HEIC"
# JPGファイルの拡張子
jpg_ext = ".jpg"                

# HEICファイルのパスを取得
heic_files = glob.glob(os.path.join(target_dir, f"*{heic_ext}"))
if not heic_files:
    print("⚠️ HEICファイルが見つかりません。")  
else:   
    for heic_file in heic_files:
        # HEICファイルの名前を取得
        base_name = os.path.basename(heic_file)
        # 新しいJPGファイルの名前を作成
        jpg_file = os.path.splitext(heic_file)[0] + jpg_ext
        
        # HEICをJPGに変換
        try:
            with Image.open(heic_file) as img:
                img.convert("RGB").save(jpg_file, "JPEG")
            print(f"{base_name} を {jpg_file} に変換しました。")
        except Exception as e:
            print(f"⚠️ {base_name} の変換中にエラーが発生しました: {e}")
        # HEICファイルを削除（必要に応じて）
        # try:
        #     os.remove(heic_file)
        #     print(f"{base_name} を削除しました。")
        except Exception as e:
            print(f"⚠️ {base_name} の削除中にエラーが発生しました: {e}")    
# HEICファイルが存在しない場合のメッセージ
if not heic_files:
    print("⚠️ HEICファイルが見つかりません。")
# HEICファイルをJPGに変換するためのスクリプト
# このスクリプトは、指定されたディレクトリ内のHEICファイルをJPG形式に変換し、変換後にHEICファイルを削除します。
# PILライブラリを使用してHEICファイルをJPGに変換します。
# 変換後のJPGファイルは、元のHEICファイルと同じディレクトリに保存されます。
# 変換に失敗した場合や、HEICファイルの削除に失敗した場合は、エラーメッセージを表示します。
# このスクリプトを実行する前に、PILライブラリがインストールされていることを確認してください。
#

