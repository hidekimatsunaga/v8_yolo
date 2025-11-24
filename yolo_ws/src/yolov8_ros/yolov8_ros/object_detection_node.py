import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from scipy.spatial import distance as dist # 距離計算のために追加
from std_msgs.msg import String          # 文字列メッセージのために追加
from std_msgs.msg import Bool   # ← フラグ用
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose # Poseも必要になります


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.bridge = CvBridge()

        # ... (モデルの読み込み部分は変更なし) ...
        # self.model = YOLO('/home/matsunaga-h/yolo_ws/model/0830_bread_bag.pt')
        self.model = YOLO('/home/matsunaga-h/yolo_ws/model/0828_bag.pt')
    

        self.latest_depth_image = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        torch.set_num_threads(1)

        # =================================================================
        # ▼▼▼【変更点】オブジェクト追跡用の変数を追加 ▼▼▼
        # =================================================================
        self.tracked_objects = {}  # {ID: {'center': (x, y), 'count': 連続検出回数, 'inactive': 未検出フレーム数}}
        self.next_object_id = 0
        self.CONSECUTIVE_THRESHOLD = 10  # 10フレーム連続で検出するためのしきい値
        self.MAX_INACTIVE_FRAMES = 100   # 100フレーム見失ったら追跡を終了
        # =================================================================

        # サブスクライバとパブリッシャ
        self.sub_rgb = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.sub_depth = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.sub_info = self.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.info_callback, 10)
        # self.pub_depth = self.create_publisher(PointStamped, '/detected_depth_points', 10)
        self.pub_detections = self.create_publisher(Detection3DArray, '/detected_objects_3d', 10)

    # ... (info_callback, depth_callback は変更なし) ...
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

    # =================================================================
    # ▼▼▼【変更点】image_callback の中身を大幅に変更 ▼▼▼
    # =================================================================
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB image conversion failed: {e}")
            return
        
        matched = {}

        # 現在のフレームで検出されたオブジェクトの中心座標をリスト化
        results = self.model(cv_image)[0]
        current_detections = []
        if results is not None and len(results.boxes) > 0:
            for box in results.boxes:
                if float(box.conf[0]) > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    # 検出結果とバウンディングボックス情報を一緒に保存
                    current_detections.append({'center': (center_x, center_y), 'box_data': box})
    # def image_callback(self, msg):
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #     except Exception as e:
    #         self.get_logger().error(f"RGB image conversion failed: {e}")
    #         return
        
    #     matched = {}
    #     results = self.model(cv_image)[0]
    #     current_detections = []

    #     if results is not None and len(results.boxes) > 0:
    #         for box in results.boxes:
    #             confidence = float(box.conf[0])
    #             class_id = int(box.cls[0])
    #             class_name = self.model.names[class_id]

    #             # ここでbottleクラス以外を除外
    #             if class_name != 'bottle':
    #                 continue

    #             if confidence > 0.4:
    #                 x1, y1, x2, y2 = map(int, box.xyxy[0])
    #                 center_x = int((x1 + x2) / 2)
    #                 center_y = int((y1 + y2) / 2)
    #                 current_detections.append({'center': (center_x, center_y), 'box_data': box})

        # 追跡中のオブジェクトがいなければ、現在の検出をすべて新規オブジェクトとして登録
        if len(self.tracked_objects) == 0:
            for det in current_detections:
                self.tracked_objects[self.next_object_id] = {'center': det['center'], 'count': 1, 'inactive': 0}
                self.next_object_id += 1
        # 追跡中のオブジェクトがある場合、現在の検出とマッチング
        else:
            tracked_ids = list(self.tracked_objects.keys())
            tracked_centers = [v['center'] for v in self.tracked_objects.values()]
            
            # 現在のフレームの検出の中心座標
            current_centers = [d['center'] for d in current_detections]

            if len(current_centers) > 0:
                # 追跡中のオブジェクトと現在の検出のマッチングを行う
                D = dist.cdist(np.array(tracked_centers), np.array(current_centers))
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                matched = {}

                used_rows = set()
                used_cols = set()

                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue
                    
                    # 距離が離れすぎている場合は同じオブジェクトと見なさない (しきい値は要調整)
                    if D[row, col] > 50:
                        continue

                    object_id = tracked_ids[row]
                    self.tracked_objects[object_id]['center'] = current_centers[col]
                    self.tracked_objects[object_id]['count'] += 1
                    self.tracked_objects[object_id]['inactive'] = 0
                    matched[col] = object_id        # ← 追加
                    used_rows.add(row)
                    used_cols.add(col)

                # マッチしなかった追跡オブジェクトは未検出カウントを増やす
                unmatched_rows = set(range(len(tracked_centers))) - used_rows
                for row in unmatched_rows:
                    object_id = tracked_ids[row]
                    self.tracked_objects[object_id]['inactive'] += 1
                    self.tracked_objects[object_id]['count'] = 0 # 連続検出をリセット

                # マッチしなかった検出は新規オブジェクトとして登録
                unmatched_cols = set(range(len(current_centers))) - used_cols
                for col in unmatched_cols:
                    self.tracked_objects[self.next_object_id] = {'center': current_centers[col], 'count': 1, 'inactive': 0}
                    self.next_object_id += 1
            else:
                # このフレームで何も検出されなかった場合、全追跡オブジェクトを未検出にする
                for object_id in self.tracked_objects:
                    self.tracked_objects[object_id]['inactive'] += 1
                    self.tracked_objects[object_id]['count'] = 0

        # 処理と描画
        # まず、長期間未検出のオブジェクトを削除
        self.tracked_objects = {oid: data for oid, data in self.tracked_objects.items() if data['inactive'] <= self.MAX_INACTIVE_FRAMES}
        
        # 追跡オブジェクトを処理するループの前に、からのDetection3DArrayメッセージを準備
        detections_msg = Detection3DArray()
        detections_msg.header.stamp = self.get_clock().now().to_msg()
        detections_msg.header.frame_id = "camera_color_optical_frame" # 座標系をヘッダーに設定

        if not matched:
            return
        for col, det in enumerate(current_detections):
                # マッチング辞書に載っていなければスキップ
            if col not in matched:
                continue
    
            oid = matched[col]
            if self.tracked_objects[oid]['count'] >= self.CONSECUTIVE_THRESHOLD:

                box = det['box_data']
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x, center_y = det['center']

                # 3次元座標の計算とパブリッシュ
                if self.latest_depth_image is not None:
                    if 0 <= center_y < self.latest_depth_image.shape[0] and 0 <= center_x < self.latest_depth_image.shape[1]:
                        z = self.latest_depth_image[center_y, center_x] * 0.001
                        if z > 0 and not np.isnan(z) and self.fx is not None:
                            X = (center_x - self.cx) * z / self.fx
                            Y = (center_y - self.cy) * z / self.fy
                            Z = z

                             # ▼▼▼ ここからが Detection3D メッセージの作成 ▼▼▼
                            detection = Detection3D()
                            detection.header = detections_msg.header # 配列のヘッダーをコピー

                            # クラスと信頼度の設定
                            hypothesis = ObjectHypothesisWithPose()
                            hypothesis.hypothesis.class_id = str(class_name) # クラス名を文字列で入れる
                            hypothesis.hypothesis.score = float(confidence)   # 信頼度
                            
                            # 3次元位置の設定
                            hypothesis.pose.pose.position.x = X
                            hypothesis.pose.pose.position.y = Y
                            hypothesis.pose.pose.position.z = Z
                            # (向きは不明なので、クォータニオンはデフォルト(0,0,0,1)のまま)
                            hypothesis.pose.pose.orientation.w = 1.0
                            
                            detection.results.append(hypothesis)

                            # 追跡IDの設定
                            detection.id = str(oid)

                            # 配列に作成した検出結果を追加
                            detections_msg.detections.append(detection)

                            # ラベルとバウンディングボックスの描画
                            label = f"ID:{oid} {class_name} ({confidence*100:.0f}%)"
                            # 検出が確定したオブジェクトは色を変えるなどすると分かりやすい
                            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2) # 赤色で表示
                            cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # ループが終わった後、検出結果が1つ以上あればパブリッシュ
        if len(detections_msg.detections) > 0:
            self.pub_detections.publish(detections_msg)
        cv2.imshow("YOLO Tracking", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()