"""
Model Export — TensorRT, ONNX, and OpenVINO export utilities.

Provides export functions for deploying YOLOv8 models in optimized
inference formats for production deployment.
"""

import os
import sys
import argparse
import logging

logger = logging.getLogger(__name__)


def export_tensorrt(
    weights: str = "models/best.pt",
    imgsz: int = 640,
    half: bool = True,
    device: int = 0,
    batch: int = 1,
    workspace: int = 4,
):
    """
    Export YOLOv8 model to TensorRT engine format.

    TensorRT provides the fastest inference on NVIDIA GPUs.
    The exported engine is specific to the GPU architecture
    it was built on (e.g., RTX 4060 = Ada Lovelace).

    Args:
        weights: Path to PyTorch model weights.
        imgsz: Inference image size.
        half: Enable FP16 half-precision.
        device: CUDA device index.
        batch: Batch size for the engine.
        workspace: TensorRT workspace size in GB.
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("🚀 TensorRT Export")
    print("=" * 60)
    print(f"  Input:     {weights}")
    print(f"  Image Size: {imgsz}")
    print(f"  FP16:      {half}")
    print(f"  Device:    cuda:{device}")
    print(f"  Batch:     {batch}")
    print(f"  Workspace: {workspace}GB")
    print("=" * 60)

    model = YOLO(weights)

    # Export to TensorRT
    engine_path = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        device=device,
        batch=batch,
        workspace=workspace,
    )

    print(f"\n✅ TensorRT engine exported: {engine_path}")
    print(f"   File size: {os.path.getsize(engine_path) / (1024**2):.1f} MB")

    # Validate
    print("\n📊 Validating exported model...")
    exported_model = YOLO(engine_path)
    print("✅ Export validation passed")

    return engine_path


def export_onnx(
    weights: str = "models/best.pt",
    imgsz: int = 640,
    half: bool = False,
    simplify: bool = True,
    opset: int = 17,
    batch: int = 1,
):
    """
    Export YOLOv8 model to ONNX format.

    ONNX provides cross-platform compatibility and can be run
    with ONNXRuntime, TensorRT, or OpenVINO backends.

    Args:
        weights: Path to PyTorch model weights.
        imgsz: Inference image size.
        half: Enable FP16 (not all ONNX runtimes support this).
        simplify: Simplify ONNX graph.
        opset: ONNX opset version.
        batch: Batch size.
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("📦 ONNX Export")
    print("=" * 60)

    model = YOLO(weights)

    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        simplify=simplify,
        opset=opset,
        batch=batch,
    )

    print(f"\n✅ ONNX model exported: {onnx_path}")
    print(f"   File size: {os.path.getsize(onnx_path) / (1024**2):.1f} MB")

    return onnx_path


def benchmark(weights: str = "models/best.pt", imgsz: int = 640, device: str = "cuda:0"):
    """
    Benchmark model in different formats.

    Tests inference speed across PyTorch, TensorRT, and ONNX formats.
    """
    from ultralytics.utils.benchmarks import benchmark as yolo_benchmark

    print("=" * 60)
    print("⚡ Model Benchmark")
    print("=" * 60)

    results = yolo_benchmark(
        model=weights,
        imgsz=imgsz,
        device=device,
    )

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model")
    parser.add_argument("--weights", type=str, default="models/best.pt",
                        help="Model weights path")
    parser.add_argument("--format", type=str, default="engine",
                        choices=["engine", "onnx", "benchmark"],
                        help="Export format")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", default=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.format == "engine":
        export_tensorrt(
            weights=args.weights,
            imgsz=args.imgsz,
            half=args.half,
            device=args.device,
            batch=args.batch,
        )
    elif args.format == "onnx":
        export_onnx(
            weights=args.weights,
            imgsz=args.imgsz,
            half=args.half,
        )
    elif args.format == "benchmark":
        benchmark(
            weights=args.weights,
            imgsz=args.imgsz,
            device=f"cuda:{args.device}",
        )
