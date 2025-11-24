# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, CameraInfo
# from geometry_msgs.msg import PointStamped
# from cv_bridge import CvBridge
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import torch
# from scipy.spatial import distance as dist # 距離計算のために追加
# from std_msgs.msg import String          # 文字列メッセージのために追加
# from std_msgs.msg import Bool   # ← フラグ用


# class YoloNode(Node):
#     def __init__(self):
#         super().__init__('yolo_node')
#         self.bridge = CvBridge()

#         # ... (モデルの読み込み部分は変更なし) ...
#         # self.model = YOLO('/home/matsunaga-h/yolo_ws/model/0830_bread_bag.pt')
#         # self.model = YOLO('/home/matsunaga-h/yolov8/yolo_ws/model/0713_300_32.pt')
#         self.model = YOLO('/home/matsunaga-h/yolov8/yolo_ws/model/0828_bag.pt')
    

#         self.latest_depth_image = None
#         self.fx = None
#         self.fy = None
#         self.cx = None
#         self.cy = None
#         torch.set_num_threads(1)

#         # 表示ウィンドウの最大サイズ（これを超える場合は縮小して表示する）
#         self.max_display_width = 960
#         self.max_display_height = 720
#         # ウィンドウを可変にしてリサイズ可能にする
#         try:
#             cv2.namedWindow("YOLO Tracking", cv2.WINDOW_NORMAL)
#         except Exception:
#             # GUI が無効な環境では例外が出る可能性があるので無視
#             pass

#         # =================================================================
#         # ▼▼▼【変更点】オブジェクト追跡用の変数を追加 ▼▼▼
#         # =================================================================
#         self.tracked_objects = {}  # {ID: {'center': (x, y), 'count': 連続検出回数, 'inactive': 未検出フレーム数}}
#         self.next_object_id = 0
#         self.CONSECUTIVE_THRESHOLD = 10  # 10フレーム連続で検出するためのしきい値
#         self.MAX_INACTIVE_FRAMES = 100   # 100フレーム見失ったら追跡を終了
#         # =================================================================

#         # サブスクライバとパブリッシャ
#         self.sub_rgb = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
#         self.sub_depth = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
#         self.sub_info = self.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.info_callback, 10)
#         self.pub_depth = self.create_publisher(PointStamped, '/detected_depth_points', 10)
#         # self.flag_pub = self.create_publisher(Bool, '/vacuum_flag', 10)
#     print("YoloNode initialized.")
#     # ... (info_callback, depth_callback は変更なし) ...
#     def info_callback(self, msg: CameraInfo):
#         self.fx = msg.k[0]
#         self.fy = msg.k[4]
#         self.cx = msg.k[2]
#         self.cy = msg.k[5]

#     def depth_callback(self, msg):
#         try:
#             self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#         except Exception as e:
#             self.get_logger().error(f"Depth image conversion failed: {e}")

#     # =================================================================
#     # ▼▼▼【変更点】image_callback の中身を大幅に変更 ▼▼▼
#     # =================================================================
#     def image_callback(self, msg):
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#         except Exception as e:
#             self.get_logger().error(f"RGB image conversion failed: {e}")
#             return

#         # 現在のフレームで検出されたオブジェクトの中心座標をリスト化
#         results = self.model(cv_image)[0]
#         current_detections = []
#         if results is not None and len(results.boxes) > 0:
#             for box in results.boxes:
#                 if float(box.conf[0]) > 0.5:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     center_x = int((x1 + x2) / 2)
#                     center_y = int((y1 + y2) / 2)
#                     current_detections.append({'center': (center_x, center_y), 'box_data': box})
                    
#         matched_cols_to_ids = {} # {current_detectionのindex: object_id}

#         # 追跡中のオブジェクトがいなければ、現在の検出をすべて新規オブジェクトとして登録
#         if len(self.tracked_objects) == 0:
#             for det in current_detections:
#                 # ▼▼▼【変更点】'locked' と 'locked_point' を追加 ▼▼▼
#                 self.tracked_objects[self.next_object_id] = {'center': det['center'], 'count': 1, 'inactive': 0, 'locked': False, 'locked_point': None}
#                 self.next_object_id += 1
#         # 追跡中のオブジェクトがある場合、現在の検出とマッチング
#         else:
#             tracked_ids = list(self.tracked_objects.keys())
#             tracked_centers = [v['center'] for v in self.tracked_objects.values()]
#             current_centers = [d['center'] for d in current_detections]


#             if len(current_centers) > 0:
#                 D = dist.cdist(np.array(tracked_centers), np.array(current_centers))
#                 rows = D.min(axis=1).argsort()
#                 cols = D.argmin(axis=1)[rows]

#                 used_rows = set()
#                 used_cols = set()

#                 for (row, col) in zip(rows, cols):
#                     if row in used_rows or col in used_cols:
#                         continue
#                     if D[row, col] > 50:
#                         continue

#                     object_id = tracked_ids[row]
                    
#                     # ▼▼▼【変更点】ロックされていなければ中心を更新 ▼▼▼
#                     if not self.tracked_objects[object_id]['locked']:
#                         self.tracked_objects[object_id]['center'] = current_centers[col]

#                     self.tracked_objects[object_id]['count'] += 1
#                     self.tracked_objects[object_id]['inactive'] = 0
#                     matched_cols_to_ids[col] = object_id
#                     used_rows.add(row)
#                     used_cols.add(col)

#                 unmatched_rows = set(range(len(tracked_centers))) - used_rows
#                 for row in unmatched_rows:
#                     object_id = tracked_ids[row]
#                     self.tracked_objects[object_id]['inactive'] += 1
#                     self.tracked_objects[object_id]['count'] = 0

#                 unmatched_cols = set(range(len(current_centers))) - used_cols
#                 for col in unmatched_cols:
#                     # ▼▼▼【変更点】'locked' と 'locked_point' を追加 ▼▼▼
#                     self.tracked_objects[self.next_object_id] = {'center': current_centers[col], 'count': 1, 'inactive': 0, 'locked': False, 'locked_point': None}
#                     self.next_object_id += 1
#             else:
#                 for object_id in self.tracked_objects:
#                     self.tracked_objects[object_id]['inactive'] += 1
#                     self.tracked_objects[object_id]['count'] = 0

#         # 長期間未検出のオブジェクトを削除
#         self.tracked_objects = {oid: data for oid, data in self.tracked_objects.items() if data['inactive'] <= self.MAX_INACTIVE_FRAMES}

#         # 処理と描画
#         for col, det in enumerate(current_detections):
#             # マッチしたオブジェクトでなければスキップ
#             if col not in matched_cols_to_ids:
#                 continue

#             oid = matched_cols_to_ids[col]
#             obj_data = self.tracked_objects[oid]
#             center_x, center_y = obj_data['center']
#             box = det['box_data']
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             class_id = int(box.cls[0])
#             class_name = self.model.names[class_id]
#             confidence = float(box.conf[0])
            
#             # 連続検出回数がしきい値を超え、まだロックされていない場合 -> ロック処理
#             if obj_data['count'] >= self.CONSECUTIVE_THRESHOLD and not obj_data['locked']:
#                 if self.latest_depth_image is not None and self.fx is not None:
#                     if 0 <= center_y < self.latest_depth_image.shape[0] and 0 <= center_x < self.latest_depth_image.shape[1]:
#                         z = self.latest_depth_image[center_y, center_x] * 0.001
#                         if z > 0 and not np.isnan(z):
#                             X = (center_x - self.cx) * z / self.fx
#                             Y = (center_y - self.cy) * z / self.fy
#                             Z = z

#                             point_msg = PointStamped()
#                             point_msg.header.stamp = self.get_clock().now().to_msg()
#                             point_msg.header.frame_id = "camera_color_optical_frame"
#                             point_msg.point.x = X
#                             point_msg.point.y = Y
#                             point_msg.point.z = Z
                            
#                             # ▼▼▼【変更点】座標を保存し、ロックする ▼▼▼
#                             self.tracked_objects[oid]['locked_point'] = point_msg
#                             self.tracked_objects[oid]['locked'] = True
#                             self.get_logger().info(f"Object ID:{oid} has been locked at X:{X:.2f}, Y:{Y:.2f}, Z:{Z:.2f}")

#             # 描画処理
#             label = f"ID:{oid} {class_name} ({confidence*100:.0f}%)"
#             color = (0, 0, 255) if obj_data['locked'] else (0, 255, 0) # ロック済みは赤、追跡中は緑
#             cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         # ▼▼▼【変更点】ロックされたオブジェクトの座標をパブリッシュし続ける ▼▼▼
#         for oid, data in self.tracked_objects.items():
#             if data['locked'] and data['locked_point'] is not None:
#                 # タイムスタンプを現在時刻に更新してパブリッシュ
#                 data['locked_point'].header.stamp = self.get_clock().now().to_msg()
#                 self.pub_depth.publish(data['locked_point'])

#         # 表示サイズを制限（大きすぎる場合は縮小）
#         try:
#             h, w = cv_image.shape[:2]
#             scale_w = self.max_display_width / float(w) if w > 0 else 1.0
#             scale_h = self.max_display_height / float(h) if h > 0 else 1.0
#             scale = min(1.0, scale_w, scale_h)
#             if scale < 1.0:
#                 disp = cv2.resize(cv_image, (int(w * scale), int(h * scale)))
#             else:
#                 disp = cv_image

#             cv2.imshow("YOLO Tracking", disp)
#             cv2.waitKey(1)
#         except Exception:
#             # 表示周りでエラーが出てもノード自体は継続させる
#             pass

# def main(args=None):
#     rclpy.init(args=args)
#     node = YoloNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from scipy.spatial import distance as dist
from std_msgs.msg import Bool

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.bridge = CvBridge()

        # =========================================================
        # ▼▼▼ キーボードで切り替えたいモデルを並べる ▼▼▼
        # =========================================================
        self.model_paths = [
            "/home/matsunaga-h/yolov8/yolo_ws/model/0828_bag.pt",
            "/home/matsunaga-h/yolov8/yolo_ws/model/0713_300_32.pt",
            "/home/matsunaga-h/yolov8/yolo_ws/model/0830_bread_bag.pt",
        ]
        self.model_index = 0
        self.model = YOLO(self.model_paths[self.model_index])
        self.get_logger().info(f"Loaded model[{self.model_index}]: {self.model_paths[self.model_index]}")
        # =========================================================

        self.latest_depth_image = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        torch.set_num_threads(1)

        self.max_display_width = 960
        self.max_display_height = 720
        try:
            cv2.namedWindow("YOLO Tracking", cv2.WINDOW_NORMAL)
        except Exception:
            pass

        # 追跡用
        self.tracked_objects = {}
        self.next_object_id = 0
        self.CONSECUTIVE_THRESHOLD = 10
        self.MAX_INACTIVE_FRAMES = 100

        # キー入力のデバウンス用（連打防止）
        self.last_key = -1

        # Subs/Pubs
        self.sub_rgb = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.sub_depth = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.sub_info = self.create_subscription(
            CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.info_callback, 10)
        self.pub_depth = self.create_publisher(PointStamped, '/detected_depth_points', 10)

        self.get_logger().info("YoloNode initialized.")

    # -----------------------------------------------------------
    # モデル切り替え関数
    # -----------------------------------------------------------
    def switch_model(self, new_index: int):
        new_index = new_index % len(self.model_paths)
        if new_index == self.model_index:
            return

        path = self.model_paths[new_index]
        try:
            self.get_logger().info(f"Switching model -> [{new_index}] {path}")
            self.model = YOLO(path)
            self.model_index = new_index

            # 追跡を一旦リセットしたいならここで消す
            self.tracked_objects.clear()
            self.next_object_id = 0

            self.get_logger().info("Model switched successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to switch model: {e}")

    # -----------------------------------------------------------
    def info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB image conversion failed: {e}")
            return

        # ===== YOLO 推論 =====
        results = self.model(cv_image)[0]
        current_detections = []
        if results is not None and len(results.boxes) > 0:
            for box in results.boxes:
                if float(box.conf[0]) > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    current_detections.append({'center': (center_x, center_y), 'box_data': box})

        matched_cols_to_ids = {}

        # ===== 追跡ロジック（ここは元のまま）=====
        if len(self.tracked_objects) == 0:
            for det in current_detections:
                self.tracked_objects[self.next_object_id] = {
                    'center': det['center'], 'count': 1, 'inactive': 0,
                    'locked': False, 'locked_point': None
                }
                self.next_object_id += 1
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
                        continue

                    object_id = tracked_ids[row]

                    if not self.tracked_objects[object_id]['locked']:
                        self.tracked_objects[object_id]['center'] = current_centers[col]

                    self.tracked_objects[object_id]['count'] += 1
                    self.tracked_objects[object_id]['inactive'] = 0
                    matched_cols_to_ids[col] = object_id
                    used_rows.add(row)
                    used_cols.add(col)

                unmatched_rows = set(range(len(tracked_centers))) - used_rows
                for row in unmatched_rows:
                    object_id = tracked_ids[row]
                    self.tracked_objects[object_id]['inactive'] += 1
                    self.tracked_objects[object_id]['count'] = 0

                unmatched_cols = set(range(len(current_centers))) - used_cols
                for col in unmatched_cols:
                    self.tracked_objects[self.next_object_id] = {
                        'center': current_centers[col], 'count': 1, 'inactive': 0,
                        'locked': False, 'locked_point': None
                    }
                    self.next_object_id += 1
            else:
                for object_id in self.tracked_objects:
                    self.tracked_objects[object_id]['inactive'] += 1
                    self.tracked_objects[object_id]['count'] = 0

        self.tracked_objects = {
            oid: data for oid, data in self.tracked_objects.items()
            if data['inactive'] <= self.MAX_INACTIVE_FRAMES
        }

        # ===== 描画・ロック等（元のまま）=====
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

            if obj_data['count'] >= self.CONSECUTIVE_THRESHOLD and not obj_data['locked']:
                if self.latest_depth_image is not None and self.fx is not None:
                    if 0 <= center_y < self.latest_depth_image.shape[0] and 0 <= center_x < self.latest_depth_image.shape[1]:
                        z = self.latest_depth_image[center_y, center_x] * 0.001
                        if z > 0 and not np.isnan(z):
                            X = (center_x - self.cx) * z / self.fx
                            Y = (center_y - self.cy) * z / self.fy
                            Z = z

                            point_msg = PointStamped()
                            point_msg.header.stamp = self.get_clock().now().to_msg()
                            point_msg.header.frame_id = "camera_color_optical_frame"
                            point_msg.point.x = X
                            point_msg.point.y = Y
                            point_msg.point.z = Z

                            self.tracked_objects[oid]['locked_point'] = point_msg
                            self.tracked_objects[oid]['locked'] = True
                            self.get_logger().info(
                                f"Object ID:{oid} locked at X:{X:.2f}, Y:{Y:.2f}, Z:{Z:.2f}"
                            )

            label = f"ID:{oid} {class_name} ({confidence*100:.0f}%)"
            color = (0, 0, 255) if obj_data['locked'] else (0, 255, 0)
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(cv_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for oid, data in self.tracked_objects.items():
            if data['locked'] and data['locked_point'] is not None:
                data['locked_point'].header.stamp = self.get_clock().now().to_msg()
                self.pub_depth.publish(data['locked_point'])

        # ===== 表示 & キー入力 =====
        try:
            h, w = cv_image.shape[:2]
            scale_w = self.max_display_width / float(w) if w > 0 else 1.0
            scale_h = self.max_display_height / float(h) if h > 0 else 1.0
            scale = min(1.0, scale_w, scale_h)
            disp = cv2.resize(cv_image, (int(w * scale), int(h * scale))) if scale < 1.0 else cv_image

            cv2.imshow("YOLO Tracking", disp)

            key = cv2.waitKey(1) & 0xFF
            if key != 255 and key != self.last_key:  # 255 は入力なし扱い
                self.last_key = key

                # 'n' で次のモデル, 'p' で前のモデル
                if key == ord('n'):
                    self.switch_model(self.model_index + 1)
                elif key == ord('p'):
                    self.switch_model(self.model_index - 1)
                # 数字キーで直接選択 (1,2,3...)
                elif ord('1') <= key <= ord(str(min(9, len(self.model_paths)))):
                    idx = key - ord('1')
                    self.switch_model(idx)

            if key == 255:
                self.last_key = -1  # 入力なしが続いたらリセット

        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
