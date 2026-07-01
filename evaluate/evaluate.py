"""
Evaluation script for pothole detection model.

Computes mAP, precision, recall on validation/test dataset.

Usage:
    python -m evaluate.evaluate --model runs/detect/pothole_run/weights/best.pt --data dataset.yaml
"""

import argparse
import logging

from ultralytics import YOLO

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate pothole detection model")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Model path to evaluate (default: yolo11n.pt)",
    )
    parser.add_argument("--data", default="dataset.yaml", help="Dataset YAML path")
    parser.add_argument("--imgsz", type=int, default=768, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    args = parser.parse_args()

    model = YOLO(args.model)

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )

    print("\n=== Evaluation Results ===")
    print(f"  mAP@0.5:     {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")
    try:
        print(f"  F1-score:     {metrics.box.f1:.4f}")
    except Exception:
        pass
    print(f"  Speed (ms):   {metrics.speed}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    main()