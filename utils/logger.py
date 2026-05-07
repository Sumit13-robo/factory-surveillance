"""
Structured Incident Logger — File, console, and JSON logging.

Provides rotating file-based logging with structured incident
records for the factory surveillance system.
"""

import os
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Optional


def setup_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    max_log_size_mb: int = 50,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure structured logging with rotating file handler.

    Args:
        log_dir: Directory for log files.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        max_log_size_mb: Maximum log file size before rotation.
        backup_count: Number of rotated log files to keep.

    Returns:
        Root logger configured for the application.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create formatters
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)-20s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Root logger
    root_logger = logging.getLogger("factory_ai")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # Rotating file handler
    log_path = os.path.join(log_dir, "factory_ai.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_log_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    root_logger.info(f"Logging initialized → {log_path} (level={level})")
    return root_logger


class IncidentLogger:
    """
    Logs structured incident records in JSON Lines format.

    Each incident is a JSON object on a separate line, enabling
    easy parsing, querying, and integration with log aggregation tools.

    Usage:
        incident_logger = IncidentLogger("logs/incidents.jsonl")
        incident_logger.log(
            event_type="ppe_violation",
            class_name="no_helmet",
            confidence=0.87,
            camera="Main Floor",
            bbox=[100, 200, 300, 400],
        )
    """

    def __init__(self, filepath: str = "logs/incidents.jsonl"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._logger = logging.getLogger("factory_ai.incidents")

    def log(
        self,
        event_type: str,
        class_name: str,
        confidence: float,
        camera: str = "unknown",
        bbox: Optional[list] = None,
        track_id: Optional[int] = None,
        zone: Optional[str] = None,
        snapshot_path: Optional[str] = None,
        extra: Optional[dict] = None,
    ):
        """
        Log a structured incident record.

        Args:
            event_type: Category (ppe_violation, hazard, unsafe_behavior, collision_risk, zone_intrusion).
            class_name: Detected class name.
            confidence: Detection confidence score.
            camera: Camera name/identifier.
            bbox: Bounding box [x1, y1, x2, y2].
            track_id: Object tracking ID.
            zone: Zone name if applicable.
            snapshot_path: Path to saved snapshot image.
            extra: Additional metadata dictionary.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "camera": camera,
            "bbox": bbox,
            "track_id": track_id,
            "zone": zone,
            "snapshot_path": snapshot_path,
        }

        if extra:
            record.update(extra)

        # Write JSON line
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self._logger.error(f"Failed to write incident log: {e}")

        # Also log to standard logger
        self._logger.info(
            f"🚨 [{event_type.upper()}] {class_name} ({confidence:.0%}) "
            f"@ {camera}" + (f" in zone '{zone}'" if zone else "")
        )
