"""
ROVER-X Dashboard Launcher — Dual-Model PPE Detection
=======================================================
Loads TWO YOLO models and starts the Factory AI dashboard:
  • Model A (Primary): Custom PPE model (helmet/mask detection)
  • Model B (Protective Equipment): HuggingFace model for full PPE
    (glove, goggles, helmet, mask, shoes + violation classes)

Usage:
    python run_dashboard.py                                       # Both models
    python run_dashboard.py --camera "http://192.168.1.50:8080/video"
    python run_dashboard.py --model models/best_mask_finetuned.pt
    python run_dashboard.py --no-model-a                          # Only Model B
    python run_dashboard.py --no-model-b                          # Only Model A (legacy)
"""

import argparse
import os
import sys
import cv2

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="ROVER-X Dashboard — Dual-Model PPE Detection")
    parser.add_argument(
        "--model", default="models/best_mask_finetuned.pt",
        help="Path to primary YOLO model (default: models/best_mask_finetuned.pt)"
    )
    parser.add_argument(
        "--model-b", default="models/protective_equipment/best.pt",
        help="Path to protective equipment model (default: models/protective_equipment/best.pt)"
    )
    parser.add_argument(
        "--model-b-conf", type=float, default=0.10,
        help="Confidence threshold for Model B (default: 0.10)"
    )
    parser.add_argument(
        "--model-b-iou", type=float, default=0.45,
        help="IoU threshold for Model B (default: 0.45)"
    )
    parser.add_argument(
        "--no-model-a", action="store_true",
        help="Disable primary PPE model (use only protective equipment model)"
    )
    parser.add_argument(
        "--no-model-b", action="store_true",
        help="Disable protective equipment model (use only primary PPE model)"
    )
    parser.add_argument(
        "--camera", default=None,
        help="Android IP Webcam URL (e.g. http://192.168.137.144:8080/video)"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Dashboard port (default: 5000)"
    )
    parser.add_argument(
        "--esp32-ip", default="192.168.137.250",
        help="ESP32 receiver IP address"
    )
    args = parser.parse_args()

    # ── Banner ──
    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║      🏭  ROVER-X — Dual-Model PPE Detection System      ║")
    print("║      Model A: Custom PPE  |  Model B: Protective Equip  ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    import torch
    import numpy as np
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.version.cuda:
        print(f"⚠️  PyTorch has CUDA {torch.version.cuda} support but GPU not detected!")
        print(f"   Try: sudo modprobe nvidia && nvidia-smi")
        print(f"   Or reboot your system to reload NVIDIA drivers.")
        print()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    model_a = None
    model_b = None

    # ── Load Model A: Primary PPE model ──
    if not args.no_model_a and os.path.exists(args.model):
        print(f"📦 Loading Model A (Primary PPE): {args.model}")
        model_a = YOLO(args.model)
        model_a.to(device)
        model_a(dummy, verbose=False, imgsz=320)
        print(f"   ✅ Model A loaded on {device.upper()} — Classes: {list(model_a.names.values())}")
    elif args.no_model_a:
        print("⏭  Model A disabled (--no-model-a)")
    else:
        print(f"⚠️  Model A not found at {args.model}, skipping...")

    # ── Load Model B: Protective Equipment model (HuggingFace) ──
    if not args.no_model_b:
        model_b_path = args.model_b
        if not os.path.exists(model_b_path):
            print(f"📥 Model B not found locally, downloading from HuggingFace...")
            try:
                from huggingface_hub import hf_hub_download
                model_b_path = hf_hub_download(
                    repo_id='keremberke/yolov8s-protective-equipment-detection',
                    filename='best.pt',
                    local_dir='models/protective_equipment'
                )
                print(f"   ✅ Downloaded to: {model_b_path}")
            except Exception as e:
                print(f"   ❌ Failed to download: {e}")
                model_b_path = None

        if model_b_path and os.path.exists(model_b_path):
            print(f"📦 Loading Model B (Protective Equipment): {model_b_path}")
            model_b = YOLO(model_b_path)
            model_b.to(device)
            model_b.overrides['conf'] = args.model_b_conf
            model_b.overrides['iou'] = args.model_b_iou
            model_b.overrides['agnostic_nms'] = False
            model_b.overrides['max_det'] = 1000
            model_b(dummy, verbose=False, imgsz=320)
            print(f"   ✅ Model B loaded on {device.upper()} — Classes: {list(model_b.names.values())}")
            print(f"   📐 Optimal imgsz=320, conf={args.model_b_conf}, iou={args.model_b_iou}")
    else:
        print("⏭  Model B disabled (--no-model-b)")

    if device == "cuda":
        print(f"\n🖥  GPU: {torch.cuda.get_device_name(0)}")

    if model_a is None and model_b is None:
        print("\n❌ No models loaded! At least one model is required.")
        sys.exit(1)

    print(f"\n✅ Models loaded: {'A' if model_a else '-'} + {'B' if model_b else '-'}")

    # Build config
    config = {
        "host": "0.0.0.0",
        "port": args.port,
        "camera": {
            "android_url": args.camera or "http://192.168.137.144:8080/video",
            "fallback": 0,
        },
        "esp32": {
            "ip": args.esp32_ip,
            "port": 8888,
        },
    }

    print(f"📡 ESP32 Receiver: {config['esp32']['ip']}:{config['esp32']['port']}")
    print(f"📹 Camera URL: {config['camera']['android_url']}")
    print(f"🌐 Dashboard: http://localhost:{args.port}")
    print("=" * 60)

    # Start dashboard
    from dashboard.app import DashboardServer
    dashboard = DashboardServer(config)

    if model_a is not None:
        dashboard.set_model(model_a)
    if model_b is not None:
        dashboard.set_model_b(model_b, conf=args.model_b_conf, iou=args.model_b_iou, imgsz=320)

    try:
        dashboard.run()  # Blocking
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        dashboard.stop()
        cv2.destroyAllWindows()
        print("Goodbye!")


if __name__ == "__main__":
    main()

