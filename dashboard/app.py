"""
Factory AI Dashboard — Self-contained Flask server.

Ultra-low-latency YOLO detection with threaded camera capture.
"""

import os
import math
import time
import random
import socket
import threading
from typing import Optional

import cv2
from flask import Flask, Response, render_template, request, jsonify


class CameraThread:
    """Threaded camera reader — always holds the LATEST frame, zero queue delay."""

    def __init__(self, source):
        self.source = source
        self.cap = None
        self.frame = None
        self.grabbed = False
        self.running = False
        self.thread = None

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False

        # Low-latency camera settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return True

    def _reader(self):
        """Continuously grab frames — always overwrites with the latest."""
        while self.running:
            grabbed, frame = self.cap.read()
            if grabbed:
                self.frame = frame
                self.grabbed = True

    def read(self):
        """Return the latest frame instantly (no I/O wait)."""
        return self.grabbed, self.frame

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()


class DashboardServer:

    def __init__(self, config: dict):
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 5000)

        cam_cfg = config.get("camera", {})
        self.android_url = cam_cfg.get(
            "android_url", "http://192.0.0.4:8080/video"
        )
        self.cam_fallback = cam_cfg.get("fallback", 0)

        esp_cfg = config.get("esp32", {})
        self.esp32_ip = esp_cfg.get("ip", "192.168.137.250")
        self.esp32_port = esp_cfg.get("port", 8888)

        self.model = None          # Primary model (existing PPE)
        self.model_b = None        # Secondary model (protective equipment from HuggingFace)
        self.model_b_conf = 0.10   # Confidence threshold for model B (lower for better detection)
        self.model_b_iou = 0.45    # IoU threshold for model B
        self.model_b_imgsz = 320   # Image size for model B (320 works best for this model)
        self.cam_thread: Optional[CameraThread] = None
        self.camera_active = False
        self.output_frame = None
        self.lock = threading.Lock()

        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._detections = []
        self._detections_lock = threading.Lock()

        # Sensor data from ESP32 (MQ2 gas — raw ADC 0-4095, DHT11 temp/humidity)
        self._sensor_data = {
            "gas": 0,
            "temperature": None,
            "humidity": None,
            "connected": False,
        }
        self._sensor_lock = threading.Lock()
        self._last_sensor_time = 0.0

        # Demo mode: show ideal values for the first 60 seconds
        self._start_time = time.time()
        self._demo_duration = 60  # seconds

        self._app: Optional[Flask] = None
        self._thread: Optional[threading.Thread] = None

    def set_model(self, model):
        self.model = model

    def set_model_b(self, model, conf=0.10, iou=0.45, imgsz=320):
        """Set the secondary protective equipment detection model."""
        self.model_b = model
        self.model_b_conf = conf
        self.model_b_iou = iou
        self.model_b_imgsz = imgsz

    # ── Camera ────────────────────────────────────────────────

    def start_camera(self):
        """Start threaded camera — tries Android first, then laptop."""
        print(f"Trying Android camera at {self.android_url}...")
        self.cam_thread = CameraThread(self.android_url)
        if self.cam_thread.start():
            print("✓ Android camera connected (threaded)!")
            return

        print("Android camera not found, trying laptop webcam...")
        self.cam_thread = CameraThread(self.cam_fallback)
        if self.cam_thread.start():
            print("✓ Laptop webcam connected (threaded)!")
        else:
            print("✗ No camera found!")

    # ── ESP32 Sensor Listener ─────────────────────────────────

    def start_sensor_listener(self):
        """Listen for sensor data from ESP32 via UDP.
        Supports two formats:
          1. Plain integer gas value: "1234"
          2. CSV with temp/humidity: "1234,25.3,60.1"
        Sent on port 5005, every 200ms.
        """
        listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_sock.bind(("0.0.0.0", 5005))  # Must match rover.cpp serverPort
        listen_sock.settimeout(2)
        print("✓ Sensor listener started on UDP port 5005")

        while True:
            try:
                data, addr = listen_sock.recvfrom(1024)
                msg = data.decode().strip()
                parts = msg.split(",")
                gas_val = int(parts[0])

                temp_val = None
                hum_val = None
                if len(parts) >= 3:
                    try:
                        temp_val = round(float(parts[1]), 1)
                        hum_val = round(float(parts[2]), 1)
                    except ValueError:
                        pass  # Keep None if parsing fails

                with self._sensor_lock:
                    self._sensor_data["gas"] = gas_val
                    self._sensor_data["connected"] = True
                    if temp_val is not None:
                        self._sensor_data["temperature"] = temp_val
                    if hum_val is not None:
                        self._sensor_data["humidity"] = hum_val
                    self._last_sensor_time = time.time()
            except socket.timeout:
                # Mark disconnected if no data for 3 seconds
                if time.time() - self._last_sensor_time > 3:
                    with self._sensor_lock:
                        self._sensor_data["connected"] = False
            except (ValueError, Exception) as e:
                pass  # Ignore malformed packets

    def send_to_esp32(self, command: str) -> bool:
        try:
            self.udp_socket.sendto(
                command.encode(), (self.esp32_ip, self.esp32_port)
            )
            return True
        except Exception as e:
            print(f"✗ ESP32 error: {e}")
            return False

    # ── Frame Generator (ultra-low-latency) ───────────────────

    def generate_frames(self):
        """Generate YOLO-annotated MJPEG frames at maximum speed."""
        self.camera_active = True
        frame_count = 0
        prev_frame = None

        while self.camera_active:
            if self.cam_thread is None or not self.cam_thread.isOpened():
                time.sleep(0.5)
                continue

            grabbed, frame = self.cam_thread.read()
            if not grabbed or frame is None:
                time.sleep(0.01)
                continue

            # Skip if frame is identical reference (no new frame yet)
            if frame is prev_frame:
                time.sleep(0.005)
                continue
            prev_frame = frame

            # Run YOLO detection (both models)
            detected_labels = []
            try:
                # ── Model A: Existing PPE model ──
                annotated_frame = frame.copy()
                if self.model is not None:
                    results_a = self.model(frame, verbose=False, imgsz=256)
                    annotated_frame = results_a[0].plot()
                    for box in results_a[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = results_a[0].names[cls_id]
                        detected_labels.append({
                            "label": label,
                            "confidence": round(conf, 2),
                            "model": "ppe_primary"
                        })

                # ── Model B: Protective Equipment model (HuggingFace) ──
                if self.model_b is not None:
                    # Try primary size first, then fallback sizes for better detection
                    results_b = None
                    for try_sz in [self.model_b_imgsz, 416, 256]:
                        r = self.model_b.predict(
                            frame, verbose=False, imgsz=try_sz,
                            conf=self.model_b_conf, iou=self.model_b_iou
                        )
                        if len(r[0].boxes) > 0:
                            results_b = r
                            break
                    if results_b is None:
                        results_b = r  # use last attempt even if 0 detections

                    # Draw Model B detections on the annotated frame
                    for box in results_b[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = results_b[0].names[cls_id]
                        detected_labels.append({
                            "label": label,
                            "confidence": round(conf, 2),
                            "model": "protective_equipment"
                        })
                        # Draw bounding box for Model B detections
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Color coding: violations in red/orange, compliant in green
                        is_violation = label.startswith("no_")
                        color = (0, 0, 255) if is_violation else (0, 255, 0)
                        thickness = 2
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                        # Label with confidence
                        tag = f"[PE] {label} {conf:.0%}"
                        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated_frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(annotated_frame, tag, (x1 + 2, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                frame_count += 1
                cv2.putText(annotated_frame, "Mode: WEB | Dual-Model",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

                # Encode JPEG — quality 50 for speed
                ret, buffer = cv2.imencode('.jpg', annotated_frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, 50])
                frame_bytes = buffer.tobytes()

                with self.lock:
                    self.output_frame = frame_bytes

                with self._detections_lock:
                    self._detections = detected_labels

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except Exception as e:
                print(f"Error: {e}")
                import traceback; traceback.print_exc()
                continue

    # ── Flask App ─────────────────────────────────────────────

    def _create_app(self) -> Flask:
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        static_dir = os.path.join(os.path.dirname(__file__), "static")

        app = Flask(__name__,
                    template_folder=template_dir,
                    static_folder=static_dir)

        @app.route("/")
        def index():
            return render_template("index.html")

        @app.route("/video_feed")
        def video_feed():
            return Response(self.generate_frames(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.route("/cmd")
        def handle_command():
            cmd = request.args.get("val", "")
            success = self.send_to_esp32(cmd)
            if success:
                return jsonify({"status": "ok", "command": cmd,
                                "message": f"Command {cmd} sent"})
            return jsonify({"status": "error",
                            "message": "Failed to send to ESP32"}), 500

        @app.route("/api/detections")
        def api_detections():
            with self._detections_lock:
                return jsonify(self._detections)

        @app.route("/api/sensors")
        def api_sensors():
            elapsed = time.time() - self._start_time
            if elapsed < self._demo_duration:
                # Demo mode: return ideal values with realistic micro-fluctuations
                t = elapsed
                demo_data = {
                    "gas": int(280 + 40 * math.sin(t * 0.3) + random.randint(-15, 15)),
                    "temperature": round(24.5 + 1.0 * math.sin(t * 0.2) + random.uniform(-0.3, 0.3), 1),
                    "humidity": round(45.0 + 3.0 * math.sin(t * 0.15) + random.uniform(-0.5, 0.5), 1),
                    "connected": True,
                    "demo": True,
                    "demo_remaining": int(self._demo_duration - elapsed),
                }
                return jsonify(demo_data)

            with self._sensor_lock:
                return jsonify(self._sensor_data)

        @app.route("/stop_camera")
        def stop_camera():
            self.camera_active = False
            if self.cam_thread:
                self.cam_thread.stop()
            return jsonify({"status": "ok", "message": "Camera stopped"})

        @app.route("/status")
        def get_status():
            return jsonify({"mode": "web",
                            "camera_active": self.camera_active,
                            "esp32_ip": self.esp32_ip})

        return app

    # ── Start / Stop ──────────────────────────────────────────

    def run(self):
        self.start_camera()
        # Start sensor listener in background
        threading.Thread(target=self.start_sensor_listener, daemon=True).start()
        self._app = self._create_app()
        self._app.run(host=self.host, port=self.port,
                      debug=False, use_reloader=False, threaded=True)

    def start(self):
        self.start_camera()
        self._app = self._create_app()
        self._thread = threading.Thread(
            target=lambda: self._app.run(
                host=self.host, port=self.port,
                debug=False, use_reloader=False, threaded=True),
            daemon=True, name="Dashboard")
        self._thread.start()
        print(f"Dashboard at http://{self.host}:{self.port}")

    def stop(self):
        self.camera_active = False
        if self.cam_thread:
            self.cam_thread.stop()
        self.udp_socket.close()
        print("Dashboard stopped.")
