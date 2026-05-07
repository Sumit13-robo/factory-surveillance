#!/usr/bin/env python3
"""
Factory AI — Dual-Model PPE Monitoring System
==============================================

Pipeline:
  1. Model A (COCO pretrained YOLOv8) → detect persons
  2. For each person → crop ROI
  3. Model B (custom best.pt) → classify PPE in crop
  4. Determine compliance (helmet + mask)
  5. Temporal smoothing → stable labels
  6. Annotate frame + show dashboard

Usage:
    python main.py                          # webcam with config defaults
    python main.py --source 0              # explicit webcam
    python main.py --source video.mp4      # video file
    python main.py --source mobile         # mobile phone camera via browser
    python main.py --no-dashboard          # headless
    python main.py --save-video out.mp4    # record output
"""

import os
import sys
import time
import signal
import argparse
import logging
from pathlib import Path

import cv2
import yaml
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.dual_model_engine import DualModelEngine
from inference.cuda_utils import CUDAManager
from inference.video_stream import VideoStream
from inference.mobile_stream import MobileCameraStream
from inference.alert_engine import AlertEngine
from inference.zone_engine import ZoneEngine
from dashboard.app import DashboardServer
from utils.logger import setup_logging, IncidentLogger
from utils.metrics import FPSCounter, InferenceTimer


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(
        description="Factory AI — Dual-Model PPE Monitoring"
    )
    p.add_argument("--config",       default="configs/settings.yaml")
    p.add_argument("--source",       default=None,   help="Webcam index or RTSP / file path")
    p.add_argument("--person-model", default=None,   help="Override person model weights")
    p.add_argument("--ppe-model",    default=None,   help="Override PPE model weights")
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--no-display",   action="store_true")
    p.add_argument("--save-video",   default=None,   help="Path to save annotated video")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
class FactoryAI:
    """Orchestrates the complete dual-model PPE monitoring pipeline."""

    def __init__(self, config: dict, args):
        self.config  = config
        self.args    = args
        self.running = False

        self.cuda_manager   = None
        self.engine         = None
        self.zone_engine    = None
        self.alert_engine   = None
        self.dashboard      = None
        self.stream         = None
        self.incident_logger = None
        self.fps_counter    = FPSCounter(window_size=60)
        self.inference_timer = InferenceTimer()
        self.video_writer   = None

    # ── Setup ──────────────────────────────────────────────────

    def setup(self):
        logger = logging.getLogger("factory_ai")

        print()
        print("╔═══════════════════════════════════════════════════════════╗")
        print("║      🏭  Factory AI — Dual-Model PPE Monitor             ║")
        print("║          Model A: COCO Person  |  Model B: PPE           ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()

        # 1. CUDA
        dev_cfg = self.config.get("device", {})
        self.cuda_manager = CUDAManager(
            gpu_id=dev_cfg.get("gpu_id", 0),
            fp16=dev_cfg.get("fp16", True),
            cudnn_benchmark=dev_cfg.get("cudnn_benchmark", True),
        )
        device = self.cuda_manager.setup()
        self.cuda_manager.start_monitoring(
            interval=self.config.get("performance", {}).get("gpu_monitor_interval", 5)
        )

        # 2. Dual-Model Engine
        dm_cfg   = self.config.get("dual_model", {})
        p_cfg    = dm_cfg.get("person_model", {})
        ppe_cfg  = dm_cfg.get("ppe_model", {})

        person_weights = (
            self.args.person_model
            or p_cfg.get("weights", "yolov8n.pt")
        )
        ppe_weights = (
            self.args.ppe_model
            or ppe_cfg.get("weights", "models/best.pt")
        )

        self.engine = DualModelEngine(
            person_weights = person_weights,
            ppe_weights    = ppe_weights,
            device         = str(device),
            fp16           = dev_cfg.get("fp16", True),
            person_conf    = p_cfg.get("confidence", 0.40),
            ppe_conf       = ppe_cfg.get("confidence", 0.30),
            person_iou     = p_cfg.get("iou_threshold", 0.45),
            ppe_iou        = ppe_cfg.get("iou_threshold", 0.40),
            imgsz          = 640,
            crop_padding   = dm_cfg.get("crop_padding", 0.10),
            smooth_window  = dm_cfg.get("smooth_window", 5),
        )

        # 3. Zone Engine
        self.zone_engine = ZoneEngine(
            zones_config=self.config.get("zones", []),
            proximity_config=self.config.get("proximity"),
        )

        # 4. Alert Engine
        self.alert_engine = AlertEngine(self.config.get("alerts", {}))
        self.alert_engine.start()

        # 5. Incident Logger
        log_cfg = self.config.get("logging", {})
        self.incident_logger = IncidentLogger(
            filepath=os.path.join(
                log_cfg.get("log_dir", "logs"), "incidents.jsonl"
            )
        )

        # 6. Video Stream — tries sources in order, first available wins
        if self.args.source is not None:
            # CLI override: use exactly what the user specified
            try:
                src = int(self.args.source)
            except ValueError:
                src = self.args.source
            logger.info(f"📹 Trying CLI source: {src}")
            self.stream = VideoStream(source=src, name="CLI Source")
            self.stream.start()
            time.sleep(2)
            if not self.stream.is_connected:
                logger.warning(f"⚠️  CLI source '{src}' failed!")
                self.stream.stop()
                self.stream = None
        else:
            # Try each configured source in order
            sources = self.config.get(
                "sources", [{"name": "Webcam", "url": 0, "enabled": True}]
            )
            for s in sources:
                if not s.get("enabled", True):
                    continue
                src_url = s["url"]
                src_name = s["name"]
                logger.info(f"📹 Trying source: {src_name} ({src_url})...")
                candidate = VideoStream(source=src_url, name=src_name)
                candidate.start()
                time.sleep(2)  # Give it time to connect
                if candidate.is_connected:
                    self.stream = candidate
                    logger.info(f"✅ Connected to: {src_name}")
                    break
                else:
                    logger.warning(f"⚠️  {src_name} not available, trying next...")
                    candidate.stop()

        if self.stream is None:
            logger.error("❌ No video source available!")
            sys.exit(1)

        # 7. Dashboard
        dash_cfg = self.config.get("dashboard", {})
        if dash_cfg.get("enabled", True) and not self.args.no_dashboard:
            self.dashboard = DashboardServer(dash_cfg)
            self.dashboard.start()

        # 8. Video Writer
        if self.args.save_video:
            logger.info(f"📹 Saving output to: {self.args.save_video}")

        logger.info("✅ All systems ready. Press Q to quit.")
        logger.info("")

    # ── Inference Loop ─────────────────────────────────────────

    def run(self):
        self.running = True
        logger = logging.getLogger("factory_ai")
        show_ppe_boxes = self.config.get("performance", {}).get("show_ppe_boxes", True)

        try:
            while self.running:
                ret, frame = self.stream.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # ── Dual-model inference ──
                with self.inference_timer:
                    persons = self.engine.detect_and_track(frame)

                self.fps_counter.tick()

                # ── Zone intrusion check (on person bboxes) ──
                person_dets = [
                    {**p, "category": "person"}
                    for p in persons
                ]
                intrusion_events = self.zone_engine.check_intrusions(person_dets)

                # ── PPE violation alerts ──
                self._fire_ppe_alerts(persons, frame)

                # ── Annotate ──
                annotated = self.engine.annotate_frame(
                    frame, persons, show_ppe_boxes=show_ppe_boxes
                )
                annotated = self.zone_engine.draw_zones(annotated)

                # ── HUD overlay ──
                fps      = self.fps_counter.get()
                latency  = self.inference_timer.last_ms
                stats    = self.engine.get_stats(persons)
                self._draw_hud(annotated, fps, latency, stats)

                # ── Dashboard ──
                if self.dashboard:
                    self.dashboard.update_frame(annotated)
                    self._update_dashboard(persons, fps, latency)

                # ── Display ──
                if not self.args.no_display:
                    h, w = annotated.shape[:2]
                    display = annotated
                    if w > 1280:
                        display = cv2.resize(
                            annotated, None, fx=1280/w, fy=1280/w
                        )
                    cv2.imshow("Factory AI — PPE Monitor", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("s"):
                        snap = f"logs/snapshots/snap_{int(time.time())}.jpg"
                        os.makedirs("logs/snapshots", exist_ok=True)
                        cv2.imwrite(snap, annotated)
                        logger.info(f"📸 Snapshot: {snap}")

                # ── Video save ──
                if self.video_writer:
                    self.video_writer.write(annotated)

        except KeyboardInterrupt:
            logger.info("\n⚡ Ctrl+C — shutting down...")
        finally:
            self.shutdown()

    # ── Alert Helpers ──────────────────────────────────────────

    def _fire_ppe_alerts(self, persons: list, frame):
        for p in persons:
            compliance = p.get("compliance", {})
            for violation in compliance.get("violations", []):
                self.alert_engine.trigger({
                    "type":       "ppe_violation",
                    "class_name": violation,
                    "confidence": 1.0,
                    "message":    f"🦺 PPE VIOLATION: {violation.upper()} — Person ID {p.get('track_id', '?')}",
                    "frame":      frame,
                    "camera":     self.stream.name,
                    "bbox":       p["bbox"],
                })
                self.incident_logger.log(
                    event_type="ppe_violation",
                    class_name=violation,
                    confidence=1.0,
                    camera=self.stream.name,
                    bbox=p["bbox"],
                    track_id=p.get("track_id"),
                )

    def _draw_hud(self, frame, fps, latency, stats):
        lines = [
            f"FPS: {fps:.1f}  Latency: {latency:.1f}ms",
            f"Persons: {stats['total_persons']}  Compliant: {stats['compliant']}  Violations: {stats['violations']}",
            f"No Helmet: {stats['no_helmet']}  No Mask: {stats['no_mask']}",
        ]
        y = 30
        for line in lines:
            cv2.putText(
                frame, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA
            )
            y += 28

    def _update_dashboard(self, persons: list, fps: float, latency: float):
        gpu = self.cuda_manager.get_stats()
        stats = self.engine.get_stats(persons)
        self.dashboard.update_stats({
            "fps":             fps,
            "inference_ms":    latency,
            "detection_count": stats["total_persons"],
            "tracking_count":  stats["total_persons"],
            "categories": {
                "compliant":  stats["compliant"],
                "violations": stats["violations"],
            },
            "alert_count": self.alert_engine.stats["total_alerts"],
            "gpu": {
                "device_name":    gpu.device_name,
                "vram_used_mb":   gpu.vram_used_mb,
                "vram_total_mb":  gpu.vram_total_mb,
                "vram_percent":   gpu.vram_percent,
                "gpu_utilization": gpu.gpu_utilization,
                "temperature":    gpu.temperature,
                "power_draw_w":   gpu.power_draw_w,
            },
        })

    # ── Shutdown ───────────────────────────────────────────────

    def shutdown(self):
        logger = logging.getLogger("factory_ai")
        logger.info("🛑 Shutting down...")
        self.running = False
        if self.stream:
            self.stream.stop()
        if self.alert_engine:
            self.alert_engine.stop()
        if self.cuda_manager:
            self.cuda_manager.stop_monitoring()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        logger.info("✅ Done. Goodbye!")


# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config)
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    log_cfg = config.get("logging", {})
    setup_logging(
        log_dir=os.path.join(PROJECT_ROOT, log_cfg.get("log_dir", "logs")),
        level=log_cfg.get("level", "INFO"),
        max_log_size_mb=log_cfg.get("max_log_size_mb", 50),
        backup_count=log_cfg.get("backup_count", 5),
    )

    os.chdir(PROJECT_ROOT)

    system = FactoryAI(config, args)
    signal.signal(signal.SIGINT, lambda s, f: setattr(system, "running", False))

    system.setup()
    system.run()


if __name__ == "__main__":
    main()
