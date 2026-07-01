# Pothole Detection

Real-time road pothole detection using YOLO, OpenCV, and Supervision.

**Status**: Masih dalam tahap penyempurnaan / refinement stage.

## Instalasi

1. Clone repo & masuk direktori:
   ```powershell
   git clone https://github.com/ndadevdev/pothole-python-yolo11n.git
   cd pothole-python-yolo11n
   ```

2. Buat dan aktifkan virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

4. Download model YOLO (otomatis di-download pertama kali):
   ```powershell
   python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
   ```

## Cara Pakai

### Deteksi real-time dari kamera
```powershell
python main.py
```

### Opsi:
| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--camera` | `0` | ID kamera lokal atau URL (contoh: `http://192.168.1.x:8080/video`) |
| `--model` | auto-resolve | Path ke model `.pt` |
| `--conf` | `0.25` | Confidence threshold |
| `--imgsz` | `1280` | Frame width |

### Keyboard (saat window deteksi aktif):
- `q` — Keluar
- `s` — Screenshot (simpan ke `screenshot_<timestamp>.jpg`)
- `r` — Toggle rekaman video

### Kamera HP via Tailscale
1. Install Tailscale di HP dan PC (login akun sama)
2. Install aplikasi "IP Webcam" di HP, nyalakan server di port `:8080`
3. Jalankan:
   ```powershell
   .\start_camera.ps1
   ```
   atau manual:
   ```powershell
   python main.py --camera http://<tailscale-ip>:8080/video
   ```

### Training model
```powershell
python -m train.train --model yolo11n.pt --data train/dataset.yaml --epochs 50
```

Low-RAM: `--imgsz 416 --batch 2 --device cpu`

GPU training via Colab: buka `notebooks/colab_train.ipynb`, Runtime → T4 GPU, Run All.

### Evaluasi model
```powershell
python -m evaluate.evaluate --model runs/detect/.../best.pt --data train/dataset.yaml
```

### Konversi dataset VOC ke YOLO
```powershell
python dataset\voc2yolo.py
```

### Melanjutkan training yang terhenti
```powershell
lanjut_training.bat
```

## Struktur Direktori
```
detector/          # Inti deteksi: detector, tracker, visualizer, recorder, severity
train/             # Training script & dataset config
evaluate/          # Evaluasi model
dataset/           # Dataset (images, annotations, labels, train/val split)
notebooks/         # Colab notebook untuk GPU training & SAM label generation
```

## Catatan
- Hanya support Windows
- Output training: `runs/detect/{name}/weights/best.pt`
- Snapshot pothole tersimpan otomatis pertama kali `tracker_id` terdeteksi di `runs/snapshots/`
- Rekaman video tersimpan di `runs/recordings/`
