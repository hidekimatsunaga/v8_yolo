#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from scipy.spatial import distance as dist  # 距離計算のため
from std_msgs.msg import Bool  # 必要なら使う用


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.bridge = CvBridge()

        # ===== YOLO モデル読み込み =====
        # self.model = YOLO('/home/matsunaga-h/yolo_ws/model/0830_bread_bag.pt')
        self.model = YOLO('/home/matsunaga-h/yolov8/yolo_ws/model/0828.pt')
        # self.model = YOLO('/home/matsunaga-h/yolov8/yolo_ws/model/0828_bag.pt')

        torch.set_num_threads(1)

        # 表示ウィンドウの最大サイズ
        self.max_display_width = 960
        self.max_display_height = 720
        try:
            cv2.namedWindow("YOLO Tracking", cv2.WINDOW_NORMAL)
        except Exception:
            pass  # GUIなし環境の場合

        # ===== OpenCV で普通のカメラを起動 (/dev/video2) =====
        # 必要なら 1,2... に変えて別のカメラを指定
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera (VideoCapture(2)).")
        else:
            # 解像度指定したいならここでセット
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.get_logger().info("Camera opened on /dev/video2.")
        # ===== オブジェクト追跡用の変数 =====
        # {ID: {'center': (x, y), 'count': 連続検出回数,
        #       'inactive': 未検出フレーム数, 'locked': bool,
        #       'locked_point': PointStamped}}
        self.tracked_objects = {}
        self.next_object_id = 0
        self.CONSECUTIVE_THRESHOLD = 10   # 10フレーム連続で検出したらロック
        self.MAX_INACTIVE_FRAMES = 100    # 100フレーム見失ったら削除

        # 3D 深度はないので、2D情報を PointStamped に入れて出す
        self.pub_depth = self.create_publisher(
            PointStamped,
            '/detected_depth_points',
            10
        )

        # 30fps 相当で回すタイマー（適宜変更）
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        self.get_logger().info("YoloNode initialized (normal camera + YOLO).")

    # ===== タイマーで定期的にカメラからフレーム取得 =====
    def timer_callback(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warning("Failed to read frame from camera.")
            return

        self.process_frame(frame)

    # ===== カメラ画像1フレーム分の処理（元の image_callback 相当） =====
    def process_frame(self, cv_image):
        # YOLO 推論
        results = self.model(cv_image)[0]
        current_detections = []

        if results is not None and len(results.boxes) > 0:
            for box in results.boxes:
                if float(box.conf[0]) > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    current_detections.append({
                        'center': (center_x, center_y),
                        'box_data': box
                    })

        matched_cols_to_ids = {}  # {current_detectionのindex: object_id}

        # ===== 追跡対象がいない場合：全部新規登録 =====
        if len(self.tracked_objects) == 0:
            for det in current_detections:
                self.tracked_objects[self.next_object_id] = {
                    'center': det['center'],
                    'count': 1,
                    'inactive': 0,
                    'locked': False,
                    'locked_point': None
                }
                self.next_object_id += 1

        # ===== 追跡対象がいる場合：距離でマッチング =====
        else:
            tracked_ids = list(self.tracked_objects.keys())
            tracked_centers = [v['center'] for v in self.tracked_objects.values()]
            current_centers = [d['center'] for d in current_detections]

            if len(current_centers) > 0:
                D = dist.cdist(np.array(tracked_centers), np.array(current_centers))
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                used_rows = set()
                used_cols = set()

                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue
                    if D[row, col] > 50:
                        # 50ピクセル以上離れてたら別物とみなす
                        continue

                    object_id = tracked_ids[row]

                    # ロックされていなければ中心を更新
                    if not self.tracked_objects[object_id]['locked']:
                        self.tracked_objects[object_id]['center'] = current_centers[col]

                    self.tracked_objects[object_id]['count'] += 1
                    self.tracked_objects[object_id]['inactive'] = 0
                    matched_cols_to_ids[col] = object_id
                    used_rows.add(row)
                    used_cols.add(col)

                # マッチしなかった既存オブジェクトは見失いカウント
                unmatched_rows = set(range(len(tracked_centers))) - used_rows
                for row in unmatched_rows:
                    object_id = tracked_ids[row]
                    self.tracked_objects[object_id]['inactive'] += 1
                    self.tracked_objects[object_id]['count'] = 0

                # マッチしなかった新規検出は新たなオブジェクトとして追加
                unmatched_cols = set(range(len(current_centers))) - used_cols
                for col in unmatched_cols:
                    self.tracked_objects[self.next_object_id] = {
                        'center': current_centers[col],
                        'count': 1,
                        'inactive': 0,
                        'locked': False,
                        'locked_point': None
                    }
                    self.next_object_id += 1
            else:
                # 今フレームで何も検出できなかった場合
                for object_id in self.tracked_objects:
                    self.tracked_objects[object_id]['inactive'] += 1
                    self.tracked_objects[object_id]['count'] = 0

        # 長期間未検出のオブジェクトを削除
        self.tracked_objects = {
            oid: data
            for oid, data in self.tracked_objects.items()
            if data['inactive'] <= self.MAX_INACTIVE_FRAMES
        }

        # ===== 描画 & ロック判定 =====
        for col, det in enumerate(current_detections):
            if col not in matched_cols_to_ids:
                continue

            oid = matched_cols_to_ids[col]
            obj_data = self.tracked_objects[oid]
            center_x, center_y = obj_data['center']
            box = det['box_data']
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])

            # 一定フレーム以上追跡できたらロック
            if obj_data['count'] >= self.CONSECUTIVE_THRESHOLD and not obj_data['locked']:
                # ★ここでは深度がないので2D座標だけを保存★
                point_msg = PointStamped()
                point_msg.header.stamp = self.get_clock().now().to_msg()
                point_msg.header.frame_id = "camera_frame"  # 適宜変更
                point_msg.point.x = float(center_x)  # 画素x
                point_msg.point.y = float(center_y)  # 画素y
                point_msg.point.z = 0.0             # 深度ないので0にしておく

                self.tracked_objects[oid]['locked_point'] = point_msg
                self.tracked_objects[oid]['locked'] = True
                self.get_logger().info(
                    f"Object ID:{oid} locked at (u,v)=({center_x}, {center_y})"
                )

            # 描画
            label = f"ID:{oid} {class_name} ({confidence*100:.0f}%)"
            color = (0, 0, 255) if obj_data['locked'] else (0, 255, 0)
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                cv_image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # ===== ロックされたオブジェクトの座標をパブリッシュし続ける =====
        for oid, data in self.tracked_objects.items():
            if data['locked'] and data['locked_point'] is not None:
                # 時刻だけ更新
                data['locked_point'].header.stamp = self.get_clock().now().to_msg()
                self.pub_depth.publish(data['locked_point'])

        # ===== 画像表示 =====
        try:
            h, w = cv_image.shape[:2]
            scale_w = self.max_display_width / float(w) if w > 0 else 1.0
            scale_h = self.max_display_height / float(h) if h > 0 else 1.0
            scale = min(1.0, scale_w, scale_h)
            if scale < 1.0:
                disp = cv2.resize(cv_image, (int(w * scale), int(h * scale)))
            else:
                disp = cv_image

            cv2.imshow("YOLO Tracking", disp)
            cv2.waitKey(1)
        except Exception:
            pass

    def destroy_node(self):
        # 終了時にカメラを解放してウィンドウを閉じる
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
