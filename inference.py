#!/usr/bin/env python3
"""
YOLOv8 PPE inference for webcam, images, folders, and videos.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def resolve_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested GPU {requested}, but CUDA is not available.")
        return f"cuda:{requested}"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is not available.")
    return requested


def load_model(weights: str, device: str, half: bool) -> YOLO:
    model = YOLO(weights)
    model.to(device)
    try:
        model.fuse()
    except Exception:
        pass

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=640, device=device, half=half, verbose=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return model


def draw_fps(frame, fps: float):
    cv2.putText(
        frame,
        f"FPS {fps:.1f}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run_stream(model: YOLO, source, args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 30)

    writer = None
    if args.save:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(args.output_dir) / "ppe_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))
        print(f"Saving video: {out_path}")

    last = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            augment=args.augment,
            max_det=args.max_det,
            verbose=False,
        )
        annotated = results[0].plot(conf=True, labels=True, boxes=True)
        now = time.perf_counter()
        fps = 1.0 / max(now - last, 1e-6)
        last = now
        draw_fps(annotated, fps)

        if writer is not None:
            writer.write(annotated)
        if args.show:
            cv2.imshow("PPE Detection", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def run_batch(model: YOLO, source: str, args: argparse.Namespace) -> None:
    model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=args.half,
        augment=args.augment,
        max_det=args.max_det,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_txt,
        project=args.output_dir,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 PPE inference")
    parser.add_argument("--weights", default="runs/ppe_train/yolov8s_ppe_industrial/weights/best.pt")
    parser.add_argument("--source", default="0", help="webcam index, image, folder, or video path")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, or 0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 on CUDA")
    parser.add_argument("--augment", action="store_true", help="Use test-time augmentation for accuracy")
    parser.add_argument("--show", action="store_true", help="Display live output window")
    parser.add_argument("--save", action="store_true", help="Save webcam/video output")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO txt predictions for image/folder inference")
    parser.add_argument("--output-dir", default="runs/ppe_infer")
    parser.add_argument("--name", default="predictions")
    args = parser.parse_args()

    args.device = select_device(args.device)
    args.half = args.device.startswith("cuda") and not args.no_half
    return args


def main() -> None:
    args = parse_args()
    model = load_model(args.weights, args.device, args.half)
    source = resolve_source(args.source)

    if isinstance(source, int):
        run_stream(model, source, args)
        return

    source_path = Path(str(source))
    if source_path.is_file() and source_path.suffix.lower() in VIDEO_SUFFIXES:
        run_stream(model, str(source_path), args)
    else:
        run_batch(model, str(source), args)


if __name__ == "__main__":
    main()
