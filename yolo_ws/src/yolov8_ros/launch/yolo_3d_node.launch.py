from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolov8_ros',
            executable='yolo_3d_node',
            name='yolo_3d_node',
            parameters=[
                # ==========================================
                # ディスプレイ設定
                # ==========================================
                {'screen_width': 1920},      # スクリーン幅（自動検出されない場合）
                {'screen_height': 1080},     # スクリーン高さ（自動検出されない場合）
                {'window_pos_x': -1},        # ウィンドウX位置（-1で自動）
                {'window_pos_y': -1},        # ウィンドウY位置（-1で自動）
                
                # ==========================================
                # 深度計測設定
                # ==========================================
                {'depth_patch_size': 7},     # 深度サンプリング領域（ピクセル）
                {'visualize_depth_patch': True},  # パッチ領域の可視化
                {'show_depth_text': True},   # 選択された深度の注釈表示
                
                # ==========================================
                # 追跡設定
                # ==========================================
                {'consecutive_threshold': 10},   # ロック判定に必要な連続検出数
                {'max_inactive_frames': 100},    # 物体削除までの非検出フレーム数
            ],
            output='screen',
        ),
    ])
