from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
from datetime import datetime

SESSION_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = BASE_DIR / "runs"
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
LOG_DIR = OUTPUT_DIR / "logs"


def grid_cell(lat: float, lon: float) -> str:
    lat_dir = "S" if lat < 0 else "N"
    lon_dir = "W" if lon < 0 else "E"
    return f"{lat_dir}{abs(lat):.2f}_{lon_dir}{abs(lon):.2f}"

CAMERA_ID = 0
MAX_CAMERAS = 5
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

def _resolve_model() -> Path:
    candidates = [
        BASE_DIR / "runs" / "detect" / "pothole_final" / "weights" / "best.pt",
        BASE_DIR / "runs" / "detect" / "pothole_v3" / "weights" / "best.pt",
        BASE_DIR / "runs" / "detect" / "pothole_light" / "weights" / "best.pt",
        BASE_DIR / "yolo11n.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

YOLO_MODEL = _resolve_model()
CONFIDENCE = 0.25
IOU = 0.5

POTHOLE_CLASS_ID = 0

GPS_POLL_INTERVAL = 1.0

SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
