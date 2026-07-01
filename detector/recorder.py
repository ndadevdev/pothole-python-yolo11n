import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from .config import SNAPSHOT_DIR, OUTPUT_DIR

log = logging.getLogger(__name__)


class Recorder:
    def __init__(self):
        self._snap_dir = SNAPSHOT_DIR
        self._snap_dir.mkdir(parents=True, exist_ok=True)
        self._seen_ids: set[int] = set()

    def update(self, frame: np.ndarray, detections: sv.Detections,
               annotated: np.ndarray | None = None) -> None:
        if len(detections) == 0:
            return

        snap = annotated if annotated is not None else frame

        for i in range(len(detections)):
            tid = detections.tracker_id[i]
            if tid is None:
                continue

            if tid not in self._seen_ids:
                self._seen_ids.add(tid)
                path = self._snap_dir / f"pothole_{tid}.jpg"
                cv2.imwrite(str(path), snap)
                log.info("Snapshot saved: %s", path)


class VideoRecorder:
    def __init__(self):
        self._writer = None
        self.recording = False

    def start(self, width: int, height: int, fps: float = 20.0):
        if self.recording:
            return
        self._record_dir = OUTPUT_DIR / "recordings"
        self._record_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(self._record_dir / f"pothole_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        self.recording = True
        log.info("Recording started: %s", path)

    def write(self, frame: np.ndarray):
        if self._writer is not None and self.recording:
            self._writer.write(frame)

    def stop(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self.recording = False
        log.info("Recording stopped")

    def toggle(self, width: int, height: int, fps: float = 20.0):
        if self.recording:
            self.stop()
        else:
            self.start(width, height, fps)
