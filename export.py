#!/usr/bin/env python3
"""
Export trained PPE YOLOv8 weights to deployment formats.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def export_pt(weights: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / weights.name
    shutil.copy2(weights, dst)
    return dst


def export_ultralytics(
    weights: Path,
    fmt: str,
    imgsz: int,
    device: str,
    half: bool,
    batch: int,
    dynamic: bool,
    simplify: bool,
    workspace: int,
) -> Path:
    model = YOLO(str(weights))
    exported = model.export(
        format=fmt,
        imgsz=imgsz,
        device=device,
        half=half,
        batch=batch,
        dynamic=dynamic,
        simplify=simplify,
        workspace=workspace,
    )
    return Path(exported)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PPE YOLOv8 model")
    parser.add_argument("--weights", default="runs/ppe_train/yolov8s_ppe_industrial/weights/best.pt")
    parser.add_argument("--format", choices=["pt", "onnx", "engine", "all"], default="all")
    parser.add_argument("--output-dir", default="exports")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--half", action="store_true", help="Enable FP16 export where supported")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shapes for ONNX")
    parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX simplification")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace in GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(weights)

    output_dir = Path(args.output_dir).resolve()
    formats = ["pt", "onnx", "engine"] if args.format == "all" else [args.format]
    exported_paths = []

    for fmt in formats:
        if fmt == "pt":
            exported = export_pt(weights, output_dir)
        else:
            exported = export_ultralytics(
                weights=weights,
                fmt=fmt,
                imgsz=args.imgsz,
                device=args.device,
                half=args.half or fmt == "engine",
                batch=args.batch,
                dynamic=args.dynamic,
                simplify=not args.no_simplify,
                workspace=args.workspace,
            )
            dst = output_dir / exported.name
            if exported.resolve() != dst.resolve():
                output_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(exported, dst)
                exported = dst
        exported_paths.append(exported)
        size_mb = exported.stat().st_size / 1024 / 1024
        print(f"Exported {fmt}: {exported} ({size_mb:.1f} MB)")

    print("Done.")


if __name__ == "__main__":
    main()
