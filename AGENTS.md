# AGENTS.md — Pothole Detection

Real-time road pothole detection (YOLO + OpenCV + Supervision). Windows-only.

## Commands
```powershell
.venv\Scripts\Activate.ps1                                # Activate venv
python main.py [--camera N|URL] [--model path] [--conf 0.25] [--imgsz 1280]
python -m train.train --model yolo11n.pt --data train/dataset.yaml --epochs 50
python -m evaluate.evaluate --model runs/detect/.../best.pt --data train/dataset.yaml
python dataset\voc2yolo.py                                # Convert VOC XML → YOLO labels
lanjut_training.bat                                       # Resume interrupted training
```

- Train output: `runs/detect/{name}/weights/best.pt`.
- Low-RAM training: `--imgsz 416 --batch 2 --device cpu`.
- GPU training: open `notebooks/colab_train.ipynb` in Colab, Runtime → T4 GPU, Run All, download `best.pt`.
- Generate seg labels from bbox: `notebooks/colab_sam_labels.ipynb` (runs MobileSAM on Drive dataset).
- `start_camera.ps1` connects via Tailscale IP to phone camera at `http://<peer>:8080/video`.

## Key files & gotchas
- `detector/config.py` — `YOLO_MODEL` auto-resolves: `pothole_final` → `pothole_v3` → `pothole_light` → `yolo11n.pt`.
- `detector/logger.py` — centralized `logging` (stdout + `runs/logs/pothole.log`). All modules use `log = logging.getLogger(__name__)`.
- `main.py` — argparse: `--camera` (device ID or IP camera URL e.g. `http://192.168.1.x:8080/video`), `--model`, `--conf`, `--imgsz`. Auto-saves snapshot first time a `tracker_id` is seen → `runs/snapshots/pothole_{id}.jpg`. Manual `s` key screenshot → CWD `screenshot_{ts}.jpg`. Keyboard: `q`=quit, `s`=screenshot, `r`=record toggle. Resizable window (`WINDOW_NORMAL`). Video → `runs/recordings/pothole_{ts}.mp4` (`mp4v`). Trackbar to switch local cameras (hidden when using URL).
- `train/train.py` — hyperparams: `patience=10`, `AdamW`, `lr0=0.001`, `lrf=0.01`, `warmup_epochs=3`, `amp=True`.
- `train/dataset.yaml` — `path: ./dataset`, `train: train/images`, `val: val/images`, `nc: 1`, `names: ["pothole"]`.
- `dataset/` — VOC XML (`annotations/`), YOLO labels (`labels/`), raw images (`images/`), plus `train/` and `val/` splits. Run `voc2yolo.py` to regenerate YOLO labels from XML (80/20 split, seed 42).
- `lanjut_training.bat` — resumes from `runs/detect/pothole_light/weights/last.pt`. Hardcoded path; venv path is wrong (uses `venv\` instead of `.venv\`).
- **No tests, CI, lint, typecheck, or pre-commit.**

## Architecture
- `detector/`: `PotholeDetector` (YOLO → `sv.Detections`, `class_id=0`) → `PotholeTracker` (`sv.ByteTrack`) → `Visualizer` (Mask fill 40% + Box + Label + Trace + severity) → `Recorder` (snapshot first-seen per `tracker_id`).
- `detector/severity.py` — shallow (<2%), medium (2-8%), deep (>8%) from bbox area % of frame.
- Threading: camera I/O thread + detection thread. Main thread: display + keyboard.
