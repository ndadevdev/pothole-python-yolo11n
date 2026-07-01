import supervision as sv


def classify(detections: sv.Detections, frame_shape: tuple[int, int, int]) -> list[str]:
    h, w = frame_shape[:2]
    frame_area = h * w
    out = []
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        area_pct = ((x2 - x1) * (y2 - y1)) / frame_area * 100
        if area_pct < 2.0:
            out.append("shallow")
        elif area_pct < 8.0:
            out.append("medium")
        else:
            out.append("deep")
    return out
