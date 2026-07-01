import numpy as np
from ultralytics import YOLO
import supervision as sv

from .config import YOLO_MODEL, CONFIDENCE, IOU, POTHOLE_CLASS_ID


class PotholeDetector:
    def __init__(self, model_path: str = YOLO_MODEL):
        self.model = YOLO(model_path)
        self.confidence = CONFIDENCE
        self.iou = IOU

    def detect(self, frame: np.ndarray) -> sv.Detections:
        results = self.model(frame, conf=self.confidence, iou=self.iou, verbose=False)[0]
        raw_masks = results.masks.data.cpu().numpy() if results.masks is not None else None
        detections = sv.Detections.from_ultralytics(results)
        keep = detections.class_id == POTHOLE_CLASS_ID
        detections = detections[keep]
        if raw_masks is not None and len(detections) > 0:
            detections.mask = raw_masks[keep].astype(bool)
        return detections
