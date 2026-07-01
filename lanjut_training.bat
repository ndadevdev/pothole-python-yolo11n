@echo off
cd /d "D:\coding\pothole"
call venv\Scripts\Activate
python -c "from ultralytics import YOLO; YOLO('runs/detect/pothole_light/weights/last.pt').train(resume=True)"
pause
