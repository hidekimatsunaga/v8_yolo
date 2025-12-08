#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String, Bool
import torch
import cv2
import time
from ultralytics import YOLO


class YoloCharNode(Node):
    def __init__(self):
        super().__init__('yolo_char_node')

        # ====== Publisher ======
        # d1: /select_d1/image_result (Int32)
        self.result_pub = self.create_publisher(Int32, '/select_d1/image_result', 10)
        # b: /select_b/enable_cross (Bool)
        self.enable_cross_pub = self.create_publisher(Bool, '/select_b/enable_cross', 10)

        # ====== Subscriber ======
        # "d1" / "b" / "false" を受けてモデル切り替え＆開始／停止
        self.sub = self.create_subscription(
            String, '/yolo/use_model', self.use_model_callback, 10
        )

        # --- カメラソース (0 = /dev/video0) ---
        self.camera_source = 0

        # --- 使用するモデル定義（モデルに応じて処理関数を割り当て） ---
        self.models = {
            'd1': {
                'model_path': '/home/matsunaga-h/yolov8/yolo_ws/model/0828.pt',
                'model': None,
                'processor': self.process_frame_d1
            },
            'b': {
                'model_path': '/home/matsunaga-h/yolov8/yolo_ws/model/0828.pt',
                'model': None,
                'processor': self.process_frame_b
            }
        }

        # --- モデルをすべてロード ---
        for key, cfg in self.models.items():
            try:
                self.get_logger().info(f'Loading model for "{key}" from {cfg["model_path"]}')
                m = YOLO(cfg['model_path'])
                try:
                    m.to('cuda:0')
                except Exception:
                    pass
                cfg['model'] = m
            except Exception as e:
                self.get_logger().error(f'Failed to load model "{key}": {e}')

        # 実行状態
        self.current_model_key = None
        self.active = False
        self.cap = None
        # ウィンドウ管理フラグ: ウィンドウはウォームアップで作成し、以降は消さない
        self.window_created = False
        self.window_names = {
            'd1': 'YOLO Char Detection (d1)',
            'b': 'YOLO Signal (b)'
        }
        # ウィンドウサイズ (width, height) - 必要ならここを変更してください
        self.window_sizes = {
            'd1': (800, 450),
            'b': (640, 360)
        }

        # d1 用のクラスマッピング
        self.class_to_int = {'a': 0, 'b': 1, 'c': 2}

        # 交差点用: 「行けそう」連続カウンタ
        self.ok_streak = 0
        # 信号色の連続フレームカウンタ（blue/red）
        self.red_counter = 0
        self.blue_counter = 0

        # ウォームアップ用フラグとタイマー
        self.warmup_mode = False
        self.warmup_timer = None

        torch.backends.cudnn.benchmark = True
        try:
            torch.cuda.set_per_process_memory_fraction(0.4, 0)
        except Exception:
            pass

        self.get_logger().info('Node initialized.')

        # 起動直後ウォームアップ（カメラ）
        self.start_warmup()

    # =====================
    # /yolo/use_model コールバック
    # =====================
    def use_model_callback(self, msg: String):
        key = msg.data.strip()
        self.get_logger().info(f'Received command: "{key}"')

        # ウォームアップ中だったらキャンセルしてから切り替え
        if self.warmup_timer is not None:
            self.warmup_timer.cancel()
            self.warmup_timer = None
            self.warmup_mode = False
            self.stop_current_model()
            self.get_logger().info('Warmup cancelled by /yolo/use_model command.')

        # 停止
        if key == 'false':
            if self.active:
                self.stop_current_model()
                self.get_logger().info('Inference stopped.')
            else:
                self.get_logger().info('Already stopped.')
            return

        # モデルキーが未知
        if key not in self.models:
            self.get_logger().warn(f'Unknown model key: "{key}"')
            return

        # すでに同じモデル
        if self.active and self.current_model_key == key:
            self.get_logger().info(f'Model "{key}" already running.')
            return

        # 別のモデル実行中なら停止
        if self.active:
            self.stop_current_model()

        # ★ カメラを開く（ここを video ではなく camera に変更）
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            self.get_logger().error(f'Failed to open camera: index={self.camera_source}')
            return

        # 必要なら解像度指定（コメント外せば有効）
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.cap = cap
        self.current_model_key = key
        self.active = True
        self.get_logger().info(f'Starting camera inference with model "{key}".')

    # =====================
    # 停止処理
    # =====================
    def stop_current_model(self):
        self.active = False
        self.current_model_key = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

        # NOTE: ウィンドウはウォームアップで作成し、
        # `/yolo/use_model` で何を受け取っても消さない仕様にするため
        # ここでは `cv2.destroyAllWindows()` を呼ばない。
        return

    def publish_enable_cross(self, can_cross_this_frame: bool):
        """
        1フレームごとの「行けそう/ダメ」の結果を受け取り、
        2回連続で can_cross_this_frame == True のときだけ True を出す。
        """
        if can_cross_this_frame:
            self.ok_streak += 1
        else:
            self.ok_streak = 0

        msg = Bool()
        msg.data = (self.ok_streak >= 2)
        self.enable_cross_pub.publish(msg)

        self.get_logger().info(
            f"[enable_cross] frame_ok={can_cross_this_frame}, "
            f"streak={self.ok_streak}, publish={msg.data}"
        )

    # =====================
    # フレーム読み取り dispatcher
    # =====================
    def process_current_frame(self):
        if not self.active or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            # カメラ用なので「Video ended」ではなく、読み取りエラーとして扱う
            self.get_logger().warn("Failed to read frame from camera. Stopping current model.")
            self.stop_current_model()
            return

        # モデル固有処理
        processor = self.models[self.current_model_key]['processor']
        processor(frame)

    # =====================
    # d1 モデル用の処理
    # =====================
    def process_frame_d1(self, frame):
        model = self.models['d1']['model']
        if model is None:
            self.get_logger().warn('Model "d1" not loaded.')
            return

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img)

        # 描画表示（ウォームアップ中でもしてOKならこのままで）
        try:
            annotated = results[0].plot()
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            # ウォームアップ中はウィンドウを作って表示する
            if self.warmup_mode:
                # ウィンドウがまだなければ作成
                if not self.window_created:
                    try:
                        # only create the 'b' window (show signal) — do not create the d1 window
                        cv2.namedWindow(self.window_names['b'], cv2.WINDOW_NORMAL)
                        # resize b window
                        w, h = self.window_sizes.get('b', (960, 540))
                        try:
                            cv2.resizeWindow(self.window_names['b'], w, h)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    self.window_created = True
                    # d1 は表示しない（imshow を行わない）
                try:
                    cv2.waitKey(1)
                except Exception:
                    pass
        except Exception:
            pass

        # ★ ウォームアップ中ならここでおしまい（トピックは出さない）
        if self.warmup_mode:
            self.get_logger().info('Warmup: d1 inference only (no publish).')
            return

        # --- ここからは通常運転 ---
        best_conf = -1.0
        best_label = None

        if hasattr(results[0], 'boxes'):
            for det in results[0].boxes:
                class_id = int(det.cls[0].item())
                conf = float(det.conf[0].item())
                if isinstance(model.names, dict):
                    name = model.names.get(class_id, str(class_id))
                else:
                    name = model.names[class_id] if class_id < len(model.names) else str(class_id)

                if name in self.class_to_int and conf > best_conf:
                    best_label = name
                    best_conf = conf

        out = -1 if best_label is None else self.class_to_int[best_label]
        msg = Int32()
        msg.data = out
        self.result_pub.publish(msg)
        self.get_logger().info(f'[d1] result={out}, best_label={best_label}, conf={best_conf:.3f}')

    # =====================
    # b モデル用の処理（信号＋車）
    # =====================
    def process_frame_b(self, frame):
        model = self.models['b']['model']
        if model is None:
            self.get_logger().warn('Model "b" not loaded.')
            return

        # フレームサイズと中心
        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width / 2.0
        center_y = frame_height / 2.0

        # 車の中央位置と大きさの閾値
        central_margin = 100      # 画面中央からの許容範囲（px）
        size_threshold = 0.02     # 画面に対するbounding box面積の割合

        scale_factor = 0.5
        can_cross_this_frame = False

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        results = model(img)
        end_time = time.time()
        self.get_logger().info(
            f"(b) {frame_height}x{frame_width}, inference: {end_time - start_time:.4f} sec"
        )

        annotated_frame = results[0].plot()
        frame_draw = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        detected_red_conf = 0.0
        detected_blue_conf = 0.0
        detected_red = False
        detected_blue = False

        # 車ブロック判定
        for detection in results[0].boxes:
            class_id = int(detection.cls[0].item())
            name = model.names[class_id]
            conf = float(detection.conf[0].item())

            if name == "car":
                box = detection.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                box_center_x = (x1 + x2) / 2.0
                box_center_y = (y1 + y2) / 2.0
                box_w = x2 - x1
                box_h = y2 - y1
                box_area_ratio = (box_w * box_h) / float(frame_width * frame_height)

                if (
                    abs(box_center_x - center_x) < central_margin
                    and abs(box_center_y - center_y) < central_margin
                    and box_area_ratio > size_threshold
                ):
                    self.get_logger().info("car blocking → no crossing")
                    can_cross_this_frame = False
                    self.red_counter = 0
                    self.blue_counter = 0
                    break

        # 信号色判定（フレーム単位で最大信頼度を取り、閾値かつ連続フレームをカウント）
        frame_detected_red = False
        frame_detected_blue = False
        frame_max_red_conf = 0.0
        frame_max_blue_conf = 0.0

        for detection in results[0].boxes:
            class_id = int(detection.cls[0].item())
            name = model.names[class_id]
            conf = float(detection.conf[0].item())

            if name == "signal_red":
                frame_detected_red = True
                frame_max_red_conf = max(frame_max_red_conf, conf)

            elif name == "signal_blue":
                frame_detected_blue = True
                frame_max_blue_conf = max(frame_max_blue_conf, conf)

        # 閾値（70%）および連続フレーム数（5フレーム）で判定
        BLUE_CONF_THRESH = 0.7
        BLUE_STREAK_REQUIRED = 5

        if frame_detected_blue and frame_max_blue_conf > BLUE_CONF_THRESH:
            self.blue_counter += 1
            self.red_counter = 0
            detected_blue = True
            detected_blue_conf = frame_max_blue_conf
        elif frame_detected_red and frame_max_red_conf > BLUE_CONF_THRESH:
            # 赤が高信頼で検出された場合は青をリセットして赤カウント
            self.red_counter += 1
            self.blue_counter = 0
            detected_red = True
            detected_red_conf = frame_max_red_conf
        else:
            # どちらも確信が持てないフレームはカウンタをリセット
            self.blue_counter = 0
            self.red_counter = 0

        self.get_logger().info(f"(b) blue_counter={self.blue_counter}, red_counter={self.red_counter}, frame_blue_conf={frame_max_blue_conf:.3f}, frame_red_conf={frame_max_red_conf:.3f}")

        # 「今フレームは行けるか」判定：青が閾値超かつ連続フレーム数に達していればOK
        if detected_blue and self.blue_counter >= BLUE_STREAK_REQUIRED and detected_blue_conf > BLUE_CONF_THRESH:
            can_cross_this_frame = True
            signal_text = 'Lets go ruby! (OK)'
            text_color = (0, 255, 0)
        else:
            can_cross_this_frame = False
            if detected_blue:
                signal_text = 'Blue detected but not stable (NG)'
            elif detected_red:
                signal_text = 'Red dominant (NG)'
            else:
                signal_text = 'No confident signal (NG)'
            text_color = (0, 0, 255)

        # 2フレーム連続OKのときだけ /select_b/enable_cross = True をpublish
        if can_cross_this_frame:
            self.ok_streak += 1
        else:
            self.ok_streak = 0

        out_msg = Bool()
        out_msg.data = (self.ok_streak >= 2)
        self.enable_cross_pub.publish(out_msg)
        self.get_logger().info(f"/select_b/enable_cross publish={out_msg.data}")

        # 画面表示
        cv2.putText(
            frame_draw, signal_text, (40, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2
        )
        small = cv2.resize(frame_draw, (0, 0), fx=scale_factor, fy=scale_factor)

        try:
            if self.window_created:
                cv2.imshow(self.window_names['b'], small)
                cv2.waitKey(1)
        except Exception:
            pass

    # =====================
    # 起動直後ウォームアップ（カメラ）
    # =====================
    def start_warmup(self):
        """起動直後に一度だけ呼ぶウォームアップ処理（カメラでd1推論）"""
        if self.active:
            self.stop_current_model()

        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            self.get_logger().error(f'Warmup: failed to open camera (index={self.camera_source})')
            return

        self.cap = cap
        self.current_model_key = 'b'   # ウォームアップに使うモデル
        self.active = True
        # ウィンドウ作成（ウォームアップで表示するため）
        if not self.window_created:
            try:
                # only create the 'b' window (show signal) — do not create the d1 window
                cv2.namedWindow(self.window_names['b'], cv2.WINDOW_NORMAL)
            except Exception:
                pass
            self.window_created = True

        self.warmup_mode = True

        self.get_logger().info('Warmup: start camera + b inference for 3 seconds.')

        # 3秒後にウォームアップ終了させるタイマー
        self.warmup_timer = self.create_timer(3.0, self.end_warmup)

    def end_warmup(self):
        """ウォームアップ時間経過時に呼ばれるコールバック"""
        self.get_logger().info('Warmup: finished. Stopping camera/inference.')
        self.warmup_mode = False

        # 一旦すべて停止
        self.stop_current_model()

        # このタイマーは一回で終了
        if self.warmup_timer is not None:
            self.warmup_timer.cancel()
            self.warmup_timer = None


def main(args=None):
    rclpy.init(args=args)
    node = YoloCharNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    try:
        node.get_logger().info('Node running.')
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.01)
            if node.active:
                node.process_current_frame()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted.")
    finally:
        node.stop_current_model()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
