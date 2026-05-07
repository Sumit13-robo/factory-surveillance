"""
Performance Metrics — mAP, precision, recall, FPS measurement.

Provides utilities for evaluating model performance and measuring
real-time inference speed for the factory surveillance system.
"""

import os
import time
import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FPSCounter:
    """
    Real-time FPS measurement using a sliding window.

    Usage:
        fps = FPSCounter(window_size=60)
        while True:
            fps.tick()
            print(f"FPS: {fps.get()}")
    """

    def __init__(self, window_size: int = 60):
        self._window = deque(maxlen=window_size)
        self._last_time = time.perf_counter()

    def tick(self):
        """Record a frame timestamp."""
        now = time.perf_counter()
        self._window.append(now - self._last_time)
        self._last_time = now

    def get(self) -> float:
        """Get current FPS as float."""
        if len(self._window) == 0:
            return 0.0
        avg_dt = sum(self._window) / len(self._window)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    def get_latency_ms(self) -> float:
        """Get average frame latency in milliseconds."""
        if len(self._window) == 0:
            return 0.0
        return (sum(self._window) / len(self._window)) * 1000


class InferenceTimer:
    """
    Context manager for measuring inference latency.

    Usage:
        timer = InferenceTimer()
        with timer:
            results = model(frame)
        print(f"Inference: {timer.last_ms:.1f}ms")
    """

    def __init__(self):
        self.last_ms: float = 0.0
        self._start: float = 0.0
        self._history = deque(maxlen=100)

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.last_ms = (time.perf_counter() - self._start) * 1000
        self._history.append(self.last_ms)

    @property
    def avg_ms(self) -> float:
        """Average inference time in ms."""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    @property
    def min_ms(self) -> float:
        return min(self._history) if self._history else 0.0

    @property
    def max_ms(self) -> float:
        return max(self._history) if self._history else 0.0


def evaluate_model(model, data_yaml: str, device: str = "cuda:0") -> dict:
    """
    Run YOLOv8 validation and return performance metrics.

    Args:
        model: Loaded YOLO model instance.
        data_yaml: Path to dataset YAML config.
        device: Device string.

    Returns:
        Dict with mAP50, mAP50-95, precision, recall, per-class AP.
    """
    logger.info("📊 Running model validation...")

    results = model.val(data=data_yaml, device=device, verbose=False)

    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "per_class_ap50": {},
    }

    # Per-class AP if available
    if hasattr(results.box, "ap50") and results.names:
        for i, ap in enumerate(results.box.ap50):
            class_name = results.names.get(i, f"class_{i}")
            metrics["per_class_ap50"][class_name] = float(ap)

    logger.info(f"   mAP@50: {metrics['mAP50']:.4f}")
    logger.info(f"   mAP@50-95: {metrics['mAP50_95']:.4f}")
    logger.info(f"   Precision: {metrics['precision']:.4f}")
    logger.info(f"   Recall: {metrics['recall']:.4f}")

    return metrics


def save_confusion_matrix(model, data_yaml: str, save_dir: str = "logs"):
    """
    Generate and save confusion matrix from validation.

    Args:
        model: YOLO model instance.
        data_yaml: Dataset YAML path.
        save_dir: Directory to save the confusion matrix plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info("📊 Generating confusion matrix...")

    results = model.val(data=data_yaml, plots=True, save_dir=save_dir)
    logger.info(f"   Confusion matrix saved to: {save_dir}")

    return results
