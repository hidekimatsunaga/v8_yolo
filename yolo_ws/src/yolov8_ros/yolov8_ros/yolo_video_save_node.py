#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO

import cv2
import numpy as np
import torch
from scipy.spatial import distance as dist

import subprocess
import re
from datetime import datetime
import os


class YoloVideoSaveNode(Node):
    def __init__(self):
        super().__init__('yolo_video_save_node')
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

        torch.set_num_threads(1)

        self.latest_depth_image = None
        self.fx = self.fy = self.cx = self.cy = None

        # ===== 動画保存関連 =====
        self.output_dir = self.declare_parameter("output_dir", "/home/matsunaga-h/yolov8/video").value
        self.fps = int(self.declare_parameter("fps", 30).value)
        os.makedirs(self.output_dir, exist_ok=True)

        self.video_writer = None
        self.is_recording = False
        self.record_frame_count = 0
        self.current_video_path = None

        self.get_logger().info(f"Video output directory: {self.output_dir}")

        # ===== 表示窓関連 =====
        self.display_width = 640
        self.display_height = 480
        self.window_margin = 20
        self.window_name = "YOLO Video Recording"
        self.max_display_width = self.display_width
        self.max_display_height = self.display_height
        self.window_position = None
        self.window_moved = False

        self.screen_width = int(self.declare_parameter("screen_width", 0).value)
        self.screen_height = int(self.declare_parameter("screen_height", 0).value)
        self.window_pos_x = int(self.declare_parameter("window_pos_x", -1).value)
        self.window_pos_y = int(self.declare_parameter("window_pos_y", -1).value)
        self.init_display_window()

        # ===== 追跡用 =====
        self.tracked_objects = {}
        self.next_object_id = 0
        self.CONSECUTIVE_THRESHOLD = int(self.declare_parameter("consecutive_threshold", 10).value)
        self.MAX_INACTIVE_FRAMES = int(self.declare_parameter("max_inactive_frames", 100).value)

        # キーデバウンス
        self.last_key = -1

        # ===== Subs =====
        self.sub_rgb = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.sub_depth = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.sub_info = self.create_subscription(
            CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.info_callback, 10)

        self.get_logger().info("YoloVideoSaveNode initialized.")

    # -----------------------------------------------------------
    # Window utilities
    # -----------------------------------------------------------
    def init_display_window(self):
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
            self.window_position = self.compute_window_position()
            self.apply_window_position()
        except Exception as e:
            self.get_logger().warn(f"Display window setup skipped: {e}")

    def apply_window_position(self):
        if self.window_moved or self.window_position is None:
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
        if self.window_pos_x >= 0 and self.window_pos_y >= 0:
            self.get_logger().info(f"Using explicit window position: x={self.window_pos_x}, y={self.window_pos_y}")
            return self.window_pos_x, self.window_pos_y

        if self.screen_width > 0 and self.screen_height > 0:
            pos_x = max(0, int(self.screen_width) - self.display_width - self.window_margin)
            pos_y = self.window_margin
            self.get_logger().info(
                f"Using screen size params ({self.screen_width}x{self.screen_height}) -> position x={pos_x}, y={pos_y}"
            )
            return pos_x, pos_y

        screen_w, screen_h = self.get_screen_size()
        if screen_w and screen_h:
            pos_x = max(0, screen_w - self.display_width - self.window_margin)
            pos_y = self.window_margin
            self.get_logger().info(
                f"Auto-detected screen size ({screen_w}x{screen_h}) -> position x={pos_x}, y={pos_y}"
            )
            return pos_x, pos_y

        self.get_logger().warn("Could not determine screen size. Window will appear in default position.")
        self.get_logger().warn("For container environments, pass: --ros-args -p screen_width:=1920 -p screen_height:=1080")
        return None

    def get_screen_size(self):
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return width, height
        except Exception:
            pass

        try:
            out = subprocess.check_output("xrandr | grep '*'", shell=True, text=True)
            match = re.search(r"(\d+)x(\d+)", out)
            if match:
                return int(match.group(1)), int(match.group(2))
        except Exception:
            pass

        return None, None

    # -----------------------------------------------------------
    # Model switch
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

            self.tracked_objects.clear()
            self.next_object_id = 0

            self.get_logger().info("Model switched successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to switch model: {e}")

    def _class_name(self, class_id: int) -> str:
        # ultralytics の names は dict のことが多いが list の場合もあるので両対応
        names = getattr(self.model, "names", None)
        if names is None:
            return str(class_id)
        if isinstance(names, dict):
            return names.get(class_id, str(class_id))
        if isinstance(names, (list, tuple)):
            return names[class_id] if 0 <= class_id < len(names) else str(class_id)
        return str(class_id)

    # -----------------------------------------------------------
    # Recording control (FIXED)
    # -----------------------------------------------------------
    def start_recording(self, frame_width: int, frame_height: int):
        """VideoWriter を開いて録画開始（is_recording はここで True にする）"""
        if self.video_writer is not None and self.video_writer.isOpened():
            self.get_logger().warn("VideoWriter already opened; skipping start.")
            self.is_recording = True
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"yolo_detection_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(output_path, fourcc, self.fps, (int(frame_width), int(frame_height)))

        if vw.isOpened():
            self.video_writer = vw
            self.is_recording = True
            self.record_frame_count = 0
            self.current_video_path = output_path
            self.get_logger().info(f"Recording started: {output_path}")
        else:
            self.get_logger().error(f"Failed to open video writer for {output_path}")
            self.video_writer = None
            self.is_recording = False

    def stop_recording(self):
        """録画停止"""
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception as e:
                self.get_logger().warn(f"VideoWriter release failed: {e}")

        if self.is_recording:
            self.get_logger().info(f"Recording stopped. Total frames: {self.record_frame_count}")

        self.video_writer = None
        self.is_recording = False
        self.record_frame_count = 0
        self.current_video_path = None

    def write_frame(self, frame):
        """フレームを書き込む（writer が開いているときだけ）"""
        if not self.is_recording:
            return
        if self.video_writer is None or (not self.video_writer.isOpened()):
            return
        try:
            self.video_writer.write(frame)
            self.record_frame_count += 1
        except Exception as e:
            self.get_logger().error(f"Failed to write frame: {e}")

    # -----------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------
    def info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB image conversion failed: {e}")
            return

        h, w = cv_image.shape[:2]

        # ★「録画中なのに writer が無い」状態を自動復旧
        if self.is_recording and (self.video_writer is None or (not self.video_writer.isOpened())):
            self.get_logger().warn("Recording flag is ON but VideoWriter is not opened. Re-opening writer.")
            self.start_recording(w, h)

        # ===== YOLO 推論 =====
        try:
            results = self.model(cv_image)[0]
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            results = None

        current_detections = []
        if results is not None and getattr(results, "boxes", None) is not None and len(results.boxes) > 0:
            for box in results.boxes:
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                if conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    current_detections.append({
                        'center': (center_x, center_y),
                        'box_data': box,
                        'confidence': conf
                    })

        # ===== 追跡 =====
        matched_cols_to_ids = {}

        if len(self.tracked_objects) == 0:
            for det in current_detections:
                self.tracked_objects[self.next_object_id] = {
                    'center': det['center'], 'count': 1, 'inactive': 0,
                    'confidence': det['confidence']
                }
                self.next_object_id += 1
        else:
            tracked_ids = list(self.tracked_objects.keys())
            tracked_centers = [v['center'] for v in self.tracked_objects.values()]
            current_centers = [d['center'] for d in current_detections]

            if len(current_centers) > 0 and len(tracked_centers) > 0:
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
                    self.tracked_objects[object_id]['center'] = current_centers[col]
                    self.tracked_objects[object_id]['count'] += 1
                    self.tracked_objects[object_id]['inactive'] = 0
                    self.tracked_objects[object_id]['confidence'] = current_detections[col]['confidence']
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
                        'confidence': current_detections[col]['confidence']
                    }
                    self.next_object_id += 1
            else:
                for object_id in list(self.tracked_objects.keys()):
                    self.tracked_objects[object_id]['inactive'] += 1
                    self.tracked_objects[object_id]['count'] = 0

        self.tracked_objects = {
            oid: data for oid, data in self.tracked_objects.items()
            if data['inactive'] <= self.MAX_INACTIVE_FRAMES
        }

        # ===== 描画 =====
        for col, det in enumerate(current_detections):
            if col not in matched_cols_to_ids:
                continue

            oid = matched_cols_to_ids[col]
            box = det['box_data']
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0]) if box.cls is not None else -1
            class_name = self._class_name(class_id)
            confidence = float(box.conf[0]) if box.conf is not None else 0.0

            label = f"ID:{oid} {class_name} ({confidence*100:.0f}%)"
            color = (0, 255, 0)
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(cv_image, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ===== 表示用リサイズ =====
        try:
            scale_w = self.max_display_width / float(w) if w > 0 else 1.0
            scale_h = self.max_display_height / float(h) if h > 0 else 1.0
            scale = min(1.0, scale_w, scale_h)
            disp = cv2.resize(cv_image, (int(w * scale), int(h * scale))) if scale < 1.0 else cv_image

            # REC overlay
            if self.is_recording:
                rec_text = f"REC: {self.record_frame_count} frames"
                cv2.putText(disp, rec_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                blink = int(self.record_frame_count / 10) % 2
                if blink:
                    cv2.circle(disp, (disp.shape[1] - 30, 30), 8, (0, 0, 255), -1)

            # ★保存は元サイズ
            self.write_frame(cv_image)

            self.apply_window_position()
            cv2.imshow(self.window_name, disp)

            key = cv2.waitKey(1) & 0xFF

            # デバウンス
            if key != 255 and key != self.last_key:
                self.last_key = key

                if key == ord('r'):
                    if not self.is_recording:
                        # ★ここでは is_recording を先に True にしない！
                        self.start_recording(w, h)
                        self.get_logger().info("═══ [REC START] ═══ Press 'r' again to stop")
                    else:
                        self.stop_recording()
                        self.get_logger().info("═══ [REC STOP] ═══ Press 'r' again to start")

                elif key == ord('n'):
                    self.switch_model(self.model_index + 1)
                elif key == ord('p'):
                    self.switch_model(self.model_index - 1)
                elif ord('1') <= key <= ord(str(min(9, len(self.model_paths)))):
                    idx = key - ord('1')
                    self.switch_model(idx)

            if key == 255:
                self.last_key = -1

        except Exception as e:
            self.get_logger().error(f"Display/recording error: {e}")

    def destroy_node(self):
        self.stop_recording()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloVideoSaveNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
