# Industrial PPE Detection with YOLOv8

This project trains and deploys a custom YOLOv8 PPE detector for industrial safety monitoring.
It follows the deployment-oriented structure of `jatin-12-2002/Industry_Safety_Detection_Using_YOLOv8`
and uses `keremberke/yolov8s-protective-equipment-detection` as the default transfer-learning
foundation.

## Target Classes

The final detector is trained for 7 classes:

```text
0 helmet
1 no_helmet
2 safety_vest
3 no_safety_vest
4 mask
5 no_mask
6 person
```

## Files

```text
training.py      Dataset preparation, transfer learning, validation, metrics, sample predictions
inference.py     Webcam, image, folder, and video inference
export.py        Export to .pt, ONNX, and TensorRT engine
dataset.yaml     Default YOLO dataset config
requirements.txt Python dependencies
```

## Dataset Layout

The pipeline supports standard YOLO-format datasets:

```text
raw_dataset/
  data.yaml
  images/
    train/
    val/
  labels/
    train/
    val/
```

It also supports common `train/valid/test` layouts. During preparation it:

- validates images and removes corrupted files
- validates YOLO annotations
- remaps common class names such as `Hardhat`, `NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`, `Mask`, `NO-Mask`, and `Person`
- creates train/validation/test splits
- optionally oversamples rare classes
- applies offline robustness augmentations for blur, brightness, contrast, noise, rotation, scale, occlusion, and horizontal flip

## Install

Use the Python environment that sees your GPU. On this machine that is:

```bash
/usr/bin/python3 -m pip install -r requirements.txt
```

Check CUDA:

```bash
/usr/bin/python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

## Train

Prepare a raw YOLO dataset and train with transfer learning from the keremberke YOLOv8s PPE model:

```bash
/usr/bin/python3 training.py \
  --source-data /path/to/raw_yolo_dataset \
  --prepared-dir datasets/ppe \
  --base-weights hf-keremberke \
  --device 0 \
  --epochs 120 \
  --imgsz 800 \
  --batch 8 \
  --balance \
  --offline-augment 1
```

Use a smaller edge model:

```bash
/usr/bin/python3 training.py --source-data /path/to/raw_yolo_dataset --base-weights yolov8n --device 0
```

Use custom weights:

```bash
/usr/bin/python3 training.py --source-data /path/to/raw_yolo_dataset --base-weights models/custom.pt --device 0
```

Prepare only, without training:

```bash
/usr/bin/python3 training.py --source-data /path/to/raw_yolo_dataset --prepare-only
```

## Evaluation Outputs

Training writes outputs under:

```text
runs/ppe_train/yolov8s_ppe_industrial/
```

Important outputs:

```text
weights/best.pt
weights/last.pt
evaluation_metrics.json
confusion_matrix.png
PR_curve.png
F1_curve.png
```

`evaluation_metrics.json` contains:

- mAP50
- mAP50-95
- precision
- recall
- F1-score

Sample validation predictions are saved in:

```text
runs/ppe_train/yolov8s_ppe_industrial_samples/
```

## Inference

Webcam:

```bash
/usr/bin/python3 inference.py --weights runs/ppe_train/yolov8s_ppe_industrial/weights/best.pt --source 0 --device auto --show
```

Image or folder:

```bash
/usr/bin/python3 inference.py --weights runs/ppe_train/yolov8s_ppe_industrial/weights/best.pt --source /path/to/images --device auto
```

Video:

```bash
/usr/bin/python3 inference.py --weights runs/ppe_train/yolov8s_ppe_industrial/weights/best.pt --source input.mp4 --device auto --save
```

Higher accuracy, lower FPS:

```bash
/usr/bin/python3 inference.py --weights runs/ppe_train/yolov8s_ppe_industrial/weights/best.pt --source 0 --augment --show
```

## Export

Export all formats:

```bash
/usr/bin/python3 export.py --weights runs/ppe_train/yolov8s_ppe_industrial/weights/best.pt --format all --device 0 --half
```

ONNX only:

```bash
/usr/bin/python3 export.py --weights runs/ppe_train/yolov8s_ppe_industrial/weights/best.pt --format onnx --dynamic
```

TensorRT only:

```bash
/usr/bin/python3 export.py --weights runs/ppe_train/yolov8s_ppe_industrial/weights/best.pt --format engine --device 0 --half
```

## Accuracy Notes

For strong real-world performance, collect data from the actual deployment cameras:

- low-light production areas
- motion-blurred workers
- small and distant workers
- partial occlusion
- multiple workers
- hard negatives such as reflective surfaces, machinery, cones, tools, and non-PPE clothing

The code improves training and inference reliability, but the final model only truly outperforms
generic PPE models after fine-tuning on representative industrial images.
