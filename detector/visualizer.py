import cv2
import numpy as np
import supervision as sv

from .severity import classify


def draw_masks(frame: np.ndarray, detections: sv.Detections, alpha: float = 0.4) -> np.ndarray:
    if detections.mask is None or len(detections) == 0:
        return frame

    h, w = frame.shape[:2]
    colors = sv.ColorPalette.DEFAULT
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(len(detections)):
        m = detections.mask[i]
        if m.ndim == 3:
            m = m.squeeze()
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors.by_idx(i).as_bgr()
        cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


class Visualizer:
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        if len(detections) == 0:
            return frame

        severities = classify(detections, frame.shape)
        labels = []
        for i in range(len(detections)):
            tid = detections.tracker_id[i] if detections.tracker_id is not None else None
            conf = detections.confidence[i] if detections.confidence is not None else 0.0
            sev = severities[i][0].upper()
            if tid is not None:
                labels.append(f"#{tid} {conf:.2f} {sev}")
            else:
                labels.append(f"{conf:.2f} {sev}")

        annotated = frame.copy()
        annotated = draw_masks(annotated, detections)
        annotated = self.box_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels=labels)
        annotated = self.trace_annotator.annotate(annotated, detections)
        return annotated
