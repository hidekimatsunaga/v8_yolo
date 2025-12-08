import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
from geometry_msgs.msg import PointStamped

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.bridge = CvBridge()
        # self.model = YOLO('/home/matsunaga-h/yolo_ws/model/best_roboflow.pt')  # 学習済みモデルへのパス
        # self.model = YOLO('/home/matsunaga-h/yolo_ws/model/best.pt')  # 学習済みモデルへのパス
        self.model = YOLO('/home/matsunaga-h/yolo_ws/model/bag_only.pt')  # 学習済みモデルへのパス

        self.latest_depth_image = None

        # RGB画像の購読
        self.sub_rgb = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10)

        # Depth画像の購読
        self.sub_depth = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        # RGB画像のパブリッシャー（オプション）
        self.pub_depth = self.create_publisher(PointStamped, '/detected_depth_points', 10)

    def depth_callback(self, msg):
        self.get_logger().info("depth_callback called")

        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def image_callback(self, msg):
        self.get_logger().info("image_callback called")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB image conversion failed: {e}")
            return

        results = self.model(cv_image)[0]
        self.get_logger().info(f"Detected {len(results.boxes)} objects")
        # 結果がNoneまたは空の場合の処理
        if results is None or len(results.boxes) == 0:
            self.get_logger().info("No objects detected.")
            return

        for box in results.boxes:
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            self.get_logger().info(f" - confidence: {conf:.2f}")   
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # z距離の取得
            z = None
            if self.latest_depth_image is not None:
                if 0 <= cy < self.latest_depth_image.shape[0] and 0 <= cx < self.latest_depth_image.shape[1]:
                    z = self.latest_depth_image[cy, cx]  # 単位はmmまたはm（カメラにより異なる）
            if z is not None:
                point_msg = PointStamped()
                point_msg.header.stamp = self.get_clock().now().to_msg()
                point_msg.header.frame_id = "camera_link"  # カメラのフレーム名
                point_msg.point.x = float(cx)
                point_msg.point.y = float(cy)
                point_msg.point.z = float(z) / 1000.0  # mm→m換算（必要なら）
            

                self.pub_depth.publish(point_msg)        

            # 描画
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0,255,0), 2)
            label = f"{z:.2f} mm" if z is not None else "z: N/A"
            cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("YOLO Detection with Depth", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
