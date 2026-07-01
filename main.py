import argparse
import logging
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from detector.config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, MAX_CAMERAS, CONFIDENCE, YOLO_MODEL
from detector.detector import PotholeDetector
from detector.logger import setup as setup_logging
from detector.tracker import PotholeTracker
from detector.visualizer import Visualizer
from detector.recorder import Recorder, VideoRecorder

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pothole Detection — real-time YOLO detection")
    p.add_argument("--camera", default=str(CAMERA_ID), help="Camera device ID or IP camera URL (default: %(default)s)")
    p.add_argument("--model", default=str(YOLO_MODEL), help="YOLO model path (default: auto-resolve)")
    p.add_argument("--conf", type=float, default=CONFIDENCE, help="Confidence threshold (default: %(default)s)")
    p.add_argument("--imgsz", type=int, default=FRAME_WIDTH, help="Frame width (default: %(default)s)")
    return p.parse_args()


def _resolve_camera_source(val: str) -> int | str:
    try:
        return int(val)
    except ValueError:
        return val


def main() -> None:
    args = parse_args()

    detector = PotholeDetector(model_path=args.model)
    tracker = PotholeTracker()
    visualizer = Visualizer()
    recorder = Recorder()
    video_recorder = VideoRecorder()

    camera_source = _resolve_camera_source(args.camera)
    is_local_cam = isinstance(camera_source, int)

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        log.error("Cannot open camera: %s", camera_source)
        sys.exit(1)
    if is_local_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.imgsz)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        active_cam: int = camera_source
    else:
        active_cam = 0

    cv2.namedWindow("Pothole Detection", cv2.WINDOW_NORMAL)
    if is_local_cam:
        cv2.createTrackbar("Camera", "Pothole Detection", camera_source, MAX_CAMERAS, lambda x: None)

    lock = threading.Lock()
    latest_frame = None
    latest_annotated = None
    detection_count = 0
    detection_fps = 0
    running = True

    def camera_loop():
        nonlocal latest_frame
        nonlocal cap
        nonlocal active_cam
        while running:
            if is_local_cam:
                cam_id = cv2.getTrackbarPos("Camera", "Pothole Detection")
                if cam_id != active_cam:
                    with lock:
                        cap.release()
                        cap = cv2.VideoCapture(cam_id)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.imgsz)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                            active_cam = cam_id
                            log.info("Switched to camera %d", active_cam)
                        else:
                            log.warning("Camera %d not available", cam_id)
                            cv2.setTrackbarPos("Camera", "Pothole Detection", active_cam)
            ret, frame = cap.read()
            if ret:
                with lock:
                    latest_frame = frame

    def detection_loop():
        nonlocal latest_annotated
        nonlocal detection_count
        nonlocal detection_fps
        last_detections = None
        fps_timer = time.time()
        fps_counter = 0
        while running:
            with lock:
                frame = latest_frame.copy() if latest_frame is not None else None
            if frame is None:
                time.sleep(0.01)
                continue

            fps_counter += 1
            if fps_counter >= 15:
                now = time.time()
                detection_fps = fps_counter / (now - fps_timer)
                fps_timer = now
                fps_counter = 0

            detections = detector.detect(frame)
            detections = tracker.update(detections)
            annotated = visualizer.annotate(frame, detections)
            recorder.update(frame, detections, annotated)
            count = len(detections) if detections is not None else 0

            with lock:
                latest_annotated = annotated
                detection_count = count

    t1 = threading.Thread(target=camera_loop, daemon=True)
    t2 = threading.Thread(target=detection_loop, daemon=True)
    t1.start()
    t2.start()
    time.sleep(0.5)

    fps_timer = time.time()
    fps_counter = 0
    display_fps = 0

    log.info("Pothole Detection started — [q] Quit  [s] Screenshot  [r] Record")

    while running:
        fps_counter += 1
        if fps_counter >= 30:
            now = time.time()
            display_fps = fps_counter / (now - fps_timer)
            fps_timer = now
            fps_counter = 0

        with lock:
            to_show = latest_annotated.copy() if latest_annotated is not None else (latest_frame.copy() if latest_frame is not None else None)
            count = detection_count
            det_fps = detection_fps

        if to_show is None:
            continue

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(to_show, now_str, (to_show.shape[1] - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(to_show, f"Potholes: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cam_label = str(camera_source) if is_local_cam else "IP Camera"
        cv2.putText(to_show, f"Cam: {cam_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(to_show, f"Display: {display_fps:.0f} FPS  |  Detect: {det_fps:.1f} FPS", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if video_recorder.recording:
            cv2.putText(to_show, "REC", (to_show.shape[1] - 80, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        video_recorder.write(to_show)
        cv2.imshow("Pothole Detection", to_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running = False
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshot_{ts}.jpg", to_show)
            log.info("Screenshot saved: screenshot_%s.jpg", ts)
        elif key == ord("r"):
            h, w = to_show.shape[:2]
            video_recorder.toggle(w, h)

    video_recorder.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    setup_logging()
    main()
