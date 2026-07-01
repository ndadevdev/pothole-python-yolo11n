"""
Training script for fine-tuning YOLO on pothole dataset.

Usage:
    python -m train.train --data dataset.yaml --epochs 50

Dataset format (YOLO):
    dataset/
    ├── train/images/
    ├── train/labels/
    ├── val/images/
    ├── val/labels/
    └── dataset.yaml
"""

import argparse
import logging

from ultralytics import YOLO

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train YOLO pothole detector")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Pre-trained model path (default: yolo11n.pt)",
    )
    parser.add_argument("--data", default="dataset.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=768, help="Input image size")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 = auto)")
    parser.add_argument("--device", default="", help="Device (cpu, 0, 0,1, ...)")
    parser.add_argument("--name", default="pothole_run", help="Experiment name")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        name=args.name,
        patience=10,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        amp=True,
    )
    log.info("Training complete. Results saved to runs/detect/%s", args.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    main()