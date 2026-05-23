"""
YOLOv8 training script for nanocrystal object detection.

The model was originally trained on Google Colab (GPU). This script removes
all Colab dependencies. Supply your own data.yaml pointing to your dataset.
See data/README.md for the expected dataset format.

Results are saved to runs/detect/<name>/ by default.

Usage:
  python src/train_yolo.py --data data.yaml
  python src/train_yolo.py --data data.yaml --model yolov8s.pt --epochs 50
"""

import argparse
from ultralytics import YOLO

# ── Hyperparameters ────────────────────────────────────────────────────────────
# These match the configuration used to produce yolo_results/train2/weights/best.pt
MODEL = "yolov8n.pt"       # nano variant; try yolov8s.pt or yolov8m.pt for more capacity
EPOCHS = 30
BATCH = 16
IMG_SIZE = 640
LR0 = 0.01                 # initial learning rate
LRF = 0.01                 # final LR = LR0 * LRF
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
WARMUP_MOMENTUM = 0.8
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on nanocrystal dataset")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data.yaml (see data/README.md for format)",
    )
    parser.add_argument("--model", default=MODEL, help="Base model weights file")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument(
        "--name", default="train", help="Run name; results saved to runs/detect/<name>/"
    )
    parser.add_argument("--device", default=None, help="Device: 'cpu', '0', '0,1', etc.")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        device=args.device,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
    )


if __name__ == "__main__":
    main()
