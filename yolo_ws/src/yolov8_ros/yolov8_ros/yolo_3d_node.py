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
import subprocess
import re

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

        self.display_width = 640
        self.display_height = 480
        self.window_margin = 20
        self.window_name = "YOLO Tracking"
        self.max_display_width = self.display_width
        self.max_display_height = self.display_height
        self.window_position = None
        self.window_moved = False

        # Optional parameters for window placement when screen size cannot be detected in container
        # Use 0 as "not set" default, -1 for window position to indicate "use auto"
        self.screen_width = self.declare_parameter("screen_width", 0).value
        self.screen_height = self.declare_parameter("screen_height", 0).value
        self.window_pos_x = self.declare_parameter("window_pos_x", -1).value
        self.window_pos_y = self.declare_parameter("window_pos_y", -1).value
        self.init_display_window()

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

    def init_display_window(self):
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
            self.window_position = self.compute_window_position()
            self.apply_window_position()
        except Exception as e:
            self.get_logger().warn(f"Display window setup skipped: {e}")

    def apply_window_position(self):
        if self.window_moved:
            return
        if self.window_position is None:
            return
        try:
            cv2.moveWindow(self.window_name, int(self.window_position[0]), int(self.window_position[1]))
            self.window_moved = True
            self.get_logger().info(
                f"Window placed at x={int(self.window_position[0])}, y={int(self.window_position[1])}"
            )
        except Exception as e:
            self.get_logger().warn(f"Window move failed: {e}")

    def compute_window_position(self):
        # Explicit position parameters override everything (use -1 as sentinel for "not set")
        if self.window_pos_x >= 0 and self.window_pos_y >= 0:
            self.get_logger().info(f"Using explicit window position: x={self.window_pos_x}, y={self.window_pos_y}")
            return self.window_pos_x, self.window_pos_y

        # Use provided screen size parameters to place at top-right (0 means "not set")
        if self.screen_width > 0 and self.screen_height > 0:
            pos_x = max(0, int(self.screen_width) - self.display_width - self.window_margin)
            pos_y = self.window_margin
            self.get_logger().info(f"Using screen size params ({self.screen_width}x{self.screen_height}) -> position x={pos_x}, y={pos_y}")
            return pos_x, pos_y

        screen_w, screen_h = self.get_screen_size()
        if screen_w and screen_h:
            pos_x = max(0, screen_w - self.display_width - self.window_margin)
            pos_y = self.window_margin
            self.get_logger().info(f"Auto-detected screen size ({screen_w}x{screen_h}) -> position x={pos_x}, y={pos_y}")
            return pos_x, pos_y

        self.get_logger().warn("Could not determine screen size. Window will appear in default position.")
        self.get_logger().warn("For container environments, pass: --ros-args -p screen_width:=1920 -p screen_height:=1080")
        return None

    def get_screen_size(self):
        try:
            import tkinter as tk  # Lazy import to avoid hard dependency
            root = tk.Tk()
            root.withdraw()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return width, height
        except Exception:
            pass

        try:
            # Fallback: parse xrandr output (avoids Tk dependency)
            out = subprocess.check_output("xrandr | grep '*'", shell=True, text=True)
            match = re.search(r"(\d+)x(\d+)", out)
            if match:
                return int(match.group(1)), int(match.group(2))
        except Exception:
            pass

        return None, None

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

            self.apply_window_position()
            cv2.imshow(self.window_name, disp)

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
