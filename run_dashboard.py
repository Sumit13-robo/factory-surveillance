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


def resolve_device(torch, requested: str = "auto") -> str:
    """Return an explicit torch/Ultralytics device string."""
    if requested == "cpu":
        return "cpu"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {requested}, but CUDA is not available to PyTorch."
            )
        return requested

    if torch.cuda.is_available():
        return "cuda:0"

    return "cpu"


def optimize_model(model, device: str, half: bool):
    """Move YOLO model to the selected device and apply safe inference optimizations."""
    model.to(device)
    try:
        model.fuse()
    except Exception:
        pass

    if device.startswith("cuda") and half:
        try:
            model.model.half()
        except Exception:
            pass


def warmup_model(model, dummy, device: str, imgsz: int, half: bool, conf: float, iou: float):
    model.predict(
        dummy,
        verbose=False,
        imgsz=imgsz,
        device=device,
        half=half,
        conf=conf,
        iou=iou,
    )


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
        "--model-a-conf", type=float, default=0.25,
        help="Confidence threshold for Model A (default: 0.25)"
    )
    parser.add_argument(
        "--model-a-iou", type=float, default=0.45,
        help="IoU threshold for Model A NMS (default: 0.45)"
    )
    parser.add_argument(
        "--model-b-conf", type=float, default=0.25,
        help="Confidence threshold for Model B (default: 0.25)"
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
        "--sensor-port", type=int, default=6000,
        help="UDP port for ESP32 sensor packets (default: 6000)"
    )
    parser.add_argument(
        "--esp32-ip", default="192.168.137.250",
        help="ESP32 receiver IP address"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Inference device: auto, cpu, cuda, cuda:0, cuda:1 (default: auto)"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference"
    )
    parser.add_argument(
        "--no-half", action="store_true",
        help="Disable FP16 inference on CUDA"
    )
    parser.add_argument(
        "--imgsz-a", type=int, default=640,
        help="Model A image size (default: 640)"
    )
    parser.add_argument(
        "--imgsz-b", type=int, default=640,
        help="Model B image size (default: 640)"
    )
    parser.add_argument(
        "--accuracy-mode", action="store_true",
        help="Enable YOLO test-time augmentation for better accuracy at lower FPS"
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=50,
        help="MJPEG stream JPEG quality, 1-100 (default: 50)"
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

    requested_device = "cpu" if args.cpu else args.device
    try:
        device = resolve_device(torch, requested_device)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        print("   Check the NVIDIA driver with: nvidia-smi")
        sys.exit(1)

    use_half = device.startswith("cuda") and not args.no_half
    if device.startswith("cuda"):
        gpu_id = int(device.split(":", 1)[1]) if ":" in device else 0
        torch.cuda.set_device(gpu_id)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"🖥  GPU selected: {torch.cuda.get_device_name(gpu_id)} ({device})")
        print(f"⚡ CUDA {torch.version.cuda} | FP16 {'ON' if use_half else 'OFF'}")
    elif torch.version.cuda:
        print(f"⚠️  PyTorch has CUDA {torch.version.cuda} support but GPU not detected!")
        print("   Try: sudo modprobe nvidia && nvidia-smi")
        print("   Or reboot your system to reload NVIDIA drivers.")
        print()
    else:
        print("⚠️  CPU inference selected. Install a CUDA-enabled PyTorch build for GPU FPS.")

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    model_a = None
    model_b = None

    # ── Load Model A: Primary PPE model ──
    if not args.no_model_a and os.path.exists(args.model):
        print(f"📦 Loading Model A (Primary PPE): {args.model}")
        model_a = YOLO(args.model)
        optimize_model(model_a, device, use_half)
        warmup_model(
            model_a,
            dummy,
            device,
            args.imgsz_a,
            use_half,
            args.model_a_conf,
            args.model_a_iou,
        )
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
            optimize_model(model_b, device, use_half)
            model_b.overrides['conf'] = args.model_b_conf
            model_b.overrides['iou'] = args.model_b_iou
            model_b.overrides['agnostic_nms'] = False
            model_b.overrides['max_det'] = 1000
            warmup_model(
                model_b,
                dummy,
                device,
                args.imgsz_b,
                use_half,
                args.model_b_conf,
                args.model_b_iou,
            )
            print(f"   ✅ Model B loaded on {device.upper()} — Classes: {list(model_b.names.values())}")
            print(f"   📐 imgsz={args.imgsz_b}, conf={args.model_b_conf}, iou={args.model_b_iou}")
    else:
        print("⏭  Model B disabled (--no-model-b)")

    if model_a is None and model_b is None:
        print("\n❌ No models loaded! At least one model is required.")
        sys.exit(1)

    print(f"\n✅ Models loaded: {'A' if model_a else '-'} + {'B' if model_b else '-'}")

    # Build config
    config = {
        "host": "0.0.0.0",
        "port": args.port,
        "camera": {
            "android_url": args.camera or "http://192.0.0.4:8080/video",
            "fallback": 0,
        },
        "esp32": {
            "ip": args.esp32_ip,
            "port": 8888,
        },
        "sensor": {
            "port": args.sensor_port,
        },
        "inference": {
            "device": device,
            "half": use_half,
            "imgsz_a": args.imgsz_a,
            "conf_a": args.model_a_conf,
            "iou_a": args.model_a_iou,
            "augment": args.accuracy_mode,
            "jpeg_quality": max(1, min(100, args.jpeg_quality)),
        },
    }

    print(f"📡 ESP32 command target: {config['esp32']['ip']}:{config['esp32']['port']}")
    print(f"📡 Sensor receiver: this laptop, UDP port {config['sensor']['port']}")
    print(f"📹 Camera URL: {config['camera']['android_url']}")
    print(f"🌐 Dashboard: http://localhost:{args.port}")
    print("=" * 60)

    # Start dashboard
    from dashboard.app import DashboardServer
    dashboard = DashboardServer(config)

    if model_a is not None:
        dashboard.set_model(model_a, conf=args.model_a_conf, iou=args.model_a_iou)
    if model_b is not None:
        dashboard.set_model_b(model_b, conf=args.model_b_conf, iou=args.model_b_iou, imgsz=args.imgsz_b)

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
