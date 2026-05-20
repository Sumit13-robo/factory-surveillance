#!/usr/bin/env python3
"""
High-accuracy YOLOv8 PPE training pipeline.

This script prepares YOLO-format datasets, remaps common industrial-safety
class names into the project target classes, fine-tunes YOLOv8, and writes
validation metrics plus sample predictions.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO


TARGET_CLASSES = [
    "helmet",
    "no_helmet",
    "safety_vest",
    "no_safety_vest",
    "mask",
    "no_mask",
    "person",
]

CLASS_ALIASES = {
    "helmet": "helmet",
    "hardhat": "helmet",
    "hard_hat": "helmet",
    "hard-hat": "helmet",
    "no_helmet": "no_helmet",
    "no-hardhat": "no_helmet",
    "no_hardhat": "no_helmet",
    "no-hard_hat": "no_helmet",
    "no_hard_hat": "no_helmet",
    "no helmet": "no_helmet",
    "mask": "mask",
    "face_mask": "mask",
    "facemask": "mask",
    "no_mask": "no_mask",
    "no-mask": "no_mask",
    "no mask": "no_mask",
    "safety_vest": "safety_vest",
    "safety vest": "safety_vest",
    "safety-vest": "safety_vest",
    "vest": "safety_vest",
    "no_safety_vest": "no_safety_vest",
    "no-safety vest": "no_safety_vest",
    "no-safety-vest": "no_safety_vest",
    "no safety vest": "no_safety_vest",
    "no_vest": "no_safety_vest",
    "no-vest": "no_safety_vest",
    "no vest": "no_safety_vest",
    "person": "person",
    "worker": "person",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PreparedItem:
    image_path: Path
    label_path: Path | None
    labels: list[list[float]]
    classes: set[int]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def read_source_names(source_root: Path) -> dict[int, str]:
    for name in ("data.yaml", "dataset.yaml"):
        cfg_path = source_root / name
        if cfg_path.exists():
            cfg = load_yaml(cfg_path)
            names = cfg.get("names", {})
            if isinstance(names, list):
                return {i: str(v) for i, v in enumerate(names)}
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
    return {i: name for i, name in enumerate(TARGET_CLASSES)}


def build_class_map(source_names: dict[int, str]) -> dict[int, int]:
    target_lookup = {name: idx for idx, name in enumerate(TARGET_CLASSES)}
    remap = {}
    for src_id, raw_name in source_names.items():
        canonical = CLASS_ALIASES.get(normalize_name(raw_name), normalize_name(raw_name))
        if canonical in target_lookup:
            remap[src_id] = target_lookup[canonical]
    return remap


def find_images(source_root: Path) -> list[Path]:
    ignored_parts = {"runs", ".git", "__pycache__"}
    return sorted(
        p
        for p in source_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and not any(part in ignored_parts for part in p.parts)
    )


def infer_label_path(image_path: Path, source_root: Path) -> Path | None:
    rel = image_path.relative_to(source_root)
    parts = list(rel.parts)

    if "images" in parts:
        idx = parts.index("images")
        label_rel = Path(*parts[:idx], "labels", *parts[idx + 1 :]).with_suffix(".txt")
        candidate = source_root / label_rel
        if candidate.exists():
            return candidate

    if image_path.parent.name == "images":
        candidate = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if candidate.exists():
            return candidate

    for candidate in (
        source_root / "labels" / f"{image_path.stem}.txt",
        image_path.parent / "labels" / f"{image_path.stem}.txt",
        image_path.with_suffix(".txt"),
    ):
        if candidate.exists():
            return candidate
    return None


def is_valid_image(path: Path) -> tuple[bool, tuple[int, int] | None]:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
        return width > 0 and height > 0, (width, height)
    except Exception:
        return False, None


def parse_label_file(label_path: Path | None, remap: dict[int, int]) -> tuple[list[list[float]], Counter]:
    stats = Counter()
    labels: list[list[float]] = []
    if label_path is None:
        stats["missing_label_files"] += 1
        return labels, stats

    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            stats["short_label_rows"] += 1
            continue
        try:
            src_cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
        except ValueError:
            stats["non_numeric_label_rows"] += 1
            continue
        if src_cls not in remap:
            stats["unknown_class_rows"] += 1
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            stats["out_of_range_boxes"] += 1
            continue
        labels.append([float(remap[src_cls]), x, y, w, h])
    return labels, stats


def yolo_to_xyxy(label: list[float], width: int, height: int) -> np.ndarray:
    _, x, y, w, h = label
    x1 = (x - w / 2.0) * width
    y1 = (y - h / 2.0) * height
    x2 = (x + w / 2.0) * width
    y2 = (y + h / 2.0) * height
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def xyxy_to_yolo(cls_id: float, box: np.ndarray, width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, width - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    y2 = float(np.clip(y2, 0, height - 1))
    bw = x2 - x1
    bh = y2 - y1
    if bw < 2 or bh < 2:
        return None
    return [cls_id, (x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height, bw / width, bh / height]


def transform_boxes(labels: list[list[float]], matrix: np.ndarray, width: int, height: int) -> list[list[float]]:
    transformed = []
    for label in labels:
        cls_id = label[0]
        x1, y1, x2, y2 = yolo_to_xyxy(label, width, height)
        corners = np.array(
            [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
            dtype=np.float32,
        )
        warped = corners @ matrix.T
        new_box = np.array(
            [warped[:, 0].min(), warped[:, 1].min(), warped[:, 0].max(), warped[:, 1].max()],
            dtype=np.float32,
        )
        converted = xyxy_to_yolo(cls_id, new_box, width, height)
        if converted is not None:
            transformed.append(converted)
    return transformed


def augment_image(image: np.ndarray, labels: list[list[float]], rng: random.Random) -> tuple[np.ndarray, list[list[float]]]:
    height, width = image.shape[:2]
    out = image.copy()
    aug_labels = [row[:] for row in labels]

    angle = rng.uniform(-8.0, 8.0)
    scale = rng.uniform(0.85, 1.15)
    tx = rng.uniform(-0.04, 0.04) * width
    ty = rng.uniform(-0.04, 0.04) * height
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[:, 2] += [tx, ty]
    out = cv2.warpAffine(out, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    aug_labels = transform_boxes(aug_labels, matrix, width, height)

    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
        for row in aug_labels:
            row[1] = 1.0 - row[1]

    alpha = rng.uniform(0.75, 1.35)
    beta = rng.uniform(-32, 32)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if rng.random() < 0.35:
        out = cv2.GaussianBlur(out, (5, 5), 0)

    if rng.random() < 0.45:
        noise = np.random.default_rng(rng.randrange(1_000_000)).normal(0, 8, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.35:
        for _ in range(rng.randint(1, 3)):
            occ_w = rng.randint(max(8, width // 20), max(12, width // 7))
            occ_h = rng.randint(max(8, height // 20), max(12, height // 7))
            x1 = rng.randint(0, max(0, width - occ_w))
            y1 = rng.randint(0, max(0, height - occ_h))
            color = rng.randint(20, 90)
            cv2.rectangle(out, (x1, y1), (x1 + occ_w, y1 + occ_h), (color, color, color), -1)

    return out, aug_labels


def write_label(path: Path, labels: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls_id, x, y, w, h in labels:
        rows.append(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def copy_item(item: PreparedItem, out_root: Path, split: str, index: int) -> tuple[Path, Path]:
    image_dst = out_root / "images" / split / f"{item.image_path.stem}_{index:06d}{item.image_path.suffix.lower()}"
    label_dst = out_root / "labels" / split / f"{item.image_path.stem}_{index:06d}.txt"
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item.image_path, image_dst)
    write_label(label_dst, item.labels)
    return image_dst, label_dst


def split_items(items: list[PreparedItem], val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[PreparedItem]]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    train_seed_items: list[PreparedItem] = []
    selected_paths: set[Path] = set()
    for cls_id in range(len(TARGET_CLASSES)):
        candidates = [item for item in shuffled if cls_id in item.classes and item.image_path not in selected_paths]
        if candidates:
            chosen = rng.choice(candidates)
            train_seed_items.append(chosen)
            selected_paths.add(chosen.image_path)

    remaining = [item for item in shuffled if item.image_path not in selected_paths]
    total = len(shuffled)
    n_test = int(round(total * test_ratio))
    n_val = int(round(total * val_ratio))
    test_items = remaining[:n_test]
    val_items = remaining[n_test : n_test + n_val]
    train_items = train_seed_items + remaining[n_test + n_val :]
    return {"train": train_items, "val": val_items, "test": test_items}


def oversample_rare_classes(train_items: list[PreparedItem], min_images: int, seed: int) -> list[PreparedItem]:
    rng = random.Random(seed)
    class_to_items: dict[int, list[PreparedItem]] = {i: [] for i in range(len(TARGET_CLASSES))}
    for item in train_items:
        for cls_id in item.classes:
            class_to_items[cls_id].append(item)

    extra = []
    for cls_id, class_items in class_to_items.items():
        if not class_items or len(class_items) >= min_images:
            continue
        needed = min_images - len(class_items)
        extra.extend(rng.choice(class_items) for _ in range(needed))
    return train_items + extra


def prepare_dataset(
    source_roots: list[Path],
    output_root: Path,
    val_split: float,
    test_split: float,
    seed: int,
    balance: bool,
    min_class_images: int,
    offline_augment: int,
) -> Path:
    output_root = output_root.resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    prepared: list[PreparedItem] = []
    source_reports = []
    for source_root in source_roots:
        source_names = read_source_names(source_root)
        class_map = build_class_map(source_names)
        if not class_map:
            raise RuntimeError(f"No matching classes found in {source_root}. Check data.yaml class names.")

        source_stats = Counter()
        source_count = 0
        for image_path in find_images(source_root):
            ok, _ = is_valid_image(image_path)
            if not ok:
                stats["corrupted_images"] += 1
                source_stats["corrupted_images"] += 1
                continue
            label_path = infer_label_path(image_path, source_root)
            labels, label_stats = parse_label_file(label_path, class_map)
            stats.update(label_stats)
            source_stats.update(label_stats)
            prepared.append(
                PreparedItem(
                    image_path=image_path,
                    label_path=label_path,
                    labels=labels,
                    classes={int(row[0]) for row in labels},
                )
            )
            source_count += 1

        source_reports.append(
            {
                "source_root": str(source_root.resolve()),
                "source_names": source_names,
                "class_map": class_map,
                "valid_images": source_count,
                "stats": dict(source_stats),
            }
        )

    if not prepared:
        raise RuntimeError("No valid images found under the provided source dataset roots")

    splits = split_items(prepared, val_split, test_split, seed)
    if balance:
        splits["train"] = oversample_rare_classes(splits["train"], min_class_images, seed)

    class_counts = {split: Counter() for split in splits}
    copied_train: list[tuple[Path, list[list[float]]]] = []
    for split, split_items_list in splits.items():
        for idx, item in enumerate(split_items_list):
            image_dst, _ = copy_item(item, output_root, split, idx)
            if split == "train":
                copied_train.append((image_dst, item.labels))
            class_counts[split].update(int(row[0]) for row in item.labels)

    rng = random.Random(seed + 17)
    aug_index = 0
    for image_path, labels in copied_train:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for _ in range(max(0, offline_augment)):
            aug_img, aug_labels = augment_image(image, labels, rng)
            aug_name = f"{image_path.stem}_aug{aug_index:06d}"
            aug_img_path = output_root / "images" / "train" / f"{aug_name}.jpg"
            aug_label_path = output_root / "labels" / "train" / f"{aug_name}.txt"
            cv2.imwrite(str(aug_img_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_label(aug_label_path, aug_labels)
            class_counts["train"].update(int(row[0]) for row in aug_labels)
            aug_index += 1

    dataset_yaml = output_root / "dataset.yaml"
    write_yaml(
        dataset_yaml,
        {
            "path": str(output_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(TARGET_CLASSES),
            "names": {i: name for i, name in enumerate(TARGET_CLASSES)},
        },
    )

    report = {
        "source_roots": [str(root.resolve()) for root in source_roots],
        "output_root": str(output_root),
        "sources": source_reports,
        "stats": dict(stats),
        "splits": {name: len(value) for name, value in splits.items()},
        "offline_augmented_images": aug_index,
        "class_counts": {
            split: {TARGET_CLASSES[k]: int(v) for k, v in counts.items()}
            for split, counts in class_counts.items()
        },
    }
    (output_root / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return dataset_yaml


def resolve_weights(base: str, output_dir: Path) -> str:
    if base == "hf-keremberke":
        local = Path("models/protective_equipment/best.pt")
        if local.exists():
            return str(local)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id="keremberke/yolov8s-protective-equipment-detection",
            filename="best.pt",
            local_dir=str(output_dir / "pretrained" / "keremberke_yolov8s"),
        )
    if base in {"yolov8n", "yolov8s", "yolov8m"}:
        return f"{base}.pt"
    return base


def metrics_to_dict(metrics) -> dict:
    box = getattr(metrics, "box", None)
    precision = float(getattr(box, "mp", 0.0) or 0.0)
    recall = float(getattr(box, "mr", 0.0) or 0.0)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    payload = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(getattr(box, "map50", 0.0) or 0.0),
        "map50_95": float(getattr(box, "map", 0.0) or 0.0),
    }
    if hasattr(metrics, "results_dict"):
        payload["raw"] = {k: float(v) for k, v in metrics.results_dict.items() if isinstance(v, (int, float))}
    return payload


def collect_sample_images(data_yaml: Path, limit: int) -> list[Path]:
    cfg = load_yaml(data_yaml)
    root = Path(cfg.get("path", data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    val_rel = cfg.get("val", "images/val")
    val_dir = root / val_rel
    images = [p for p in val_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(images)[:limit]


def train(args: argparse.Namespace) -> Path:
    output_dir = Path(args.project).resolve()
    data_yaml = Path(args.data).resolve()

    if args.source_data:
        data_yaml = prepare_dataset(
            source_roots=[Path(path).resolve() for path in args.source_data],
            output_root=Path(args.prepared_dir).resolve(),
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed,
            balance=args.balance,
            min_class_images=args.min_class_images,
            offline_augment=args.offline_augment,
        )
        print(f"Prepared dataset: {data_yaml}")

    if args.prepare_only:
        return data_yaml

    weights = resolve_weights(args.base_weights, output_dir)
    model = YOLO(weights)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        amp=not args.no_amp,
        val=True,
        save=True,
        save_period=args.save_period,
        cache=args.cache,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        optimizer=args.optimizer,
        cos_lr=True,
        close_mosaic=args.close_mosaic,
        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.45,
        degrees=8.0,
        translate=0.10,
        scale=0.60,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.80,
        mixup=0.10,
        copy_paste=0.15,
        erasing=0.25,
        multi_scale=0.5,
    )

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    if not args.no_eval:
        eval_metrics = YOLO(str(best_weights)).val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            half=not args.no_amp,
            plots=True,
            save_json=True,
            project=args.project,
            name=f"{args.name}_eval",
            exist_ok=True,
        )
        metrics_payload = metrics_to_dict(eval_metrics)
        (save_dir / "evaluation_metrics.json").write_text(
            json.dumps(metrics_payload, indent=2),
            encoding="utf-8",
        )

        samples = collect_sample_images(data_yaml, args.sample_count)
        if samples:
            YOLO(str(best_weights)).predict(
                source=[str(p) for p in samples],
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                half=not args.no_amp,
                save=True,
                project=args.project,
                name=f"{args.name}_samples",
                exist_ok=True,
            )

    print(f"Best weights: {best_weights}")
    return best_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a robust YOLOv8 PPE detector")
    parser.add_argument("--source-data", nargs="+", default=None,
                        help="One or more YOLO dataset roots to validate, remap, split, and prepare")
    parser.add_argument("--prepared-dir", default="datasets/ppe", help="Prepared YOLO dataset output directory")
    parser.add_argument("--data", default="dataset.yaml", help="YOLO dataset YAML for training")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare dataset and exit")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.05)
    parser.add_argument("--balance", action="store_true", help="Oversample rare training classes")
    parser.add_argument("--min-class-images", type=int, default=400)
    parser.add_argument("--offline-augment", type=int, default=1, help="Augmented training copies per image")

    parser.add_argument("--base-weights", default="hf-keremberke",
                        help="hf-keremberke, yolov8n, yolov8s, yolov8m, or custom .pt path")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="Ultralytics device string, e.g. 0, cpu, cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")

    parser.add_argument("--project", default="runs/ppe_train")
    parser.add_argument("--name", default="yolov8s_ppe_industrial")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
