# Fabric Defect Detector v2

Real-time fabric defect detection using MobileNetV2 (TFLite / H5).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Desktop App (OpenCV window)

```bash
# Live webcam — TFLite (fastest)
python app.py

# Live webcam — H5 model
python app.py --model h5

# Single image
python app.py --source image --path path/to/image.jpg

# Video file
python app.py --source video --path path/to/video.mp4

# Override performance settings via CLI
python app.py --model h5 --tta 1 --skip 3
```

**Keyboard shortcuts:** `q` quit | `r` report | `s` snapshot | `p` pause

### Web Dashboard (Flask)

```bash
python web_app.py
python web_app.py --model h5 --port 5000
```

Then open http://localhost:5000

## Performance Tuning (`config.py`)

| Setting | Default | Effect |
|---------|---------|--------|
| `TTA_PASSES` | 1 | 1=fastest, 3=most accurate |
| `INFERENCE_SKIP_FRAMES` | 2 | Run inference every N frames |
| `ENABLE_LOCALISATION` | False | 3×3 grid bbox (slower) |
| `ENABLE_EMA_SMOOTHING` | True | Smooth predictions across frames |
| `EMA_ALPHA` | 0.5 | 0=very smooth, 1=raw |
| `ENABLE_CLAHE` | False | Brightness normalisation |

## Anti-Flicker Camera Settings (`config.py`)

| Setting | Default | Effect |
|---------|---------|--------|
| `CAM_AUTO_EXPOSURE` | True | Set False + tune `CAM_EXPOSURE` to lock |
| `CAM_EXPOSURE` | -6 | Manual exposure value |
| `CAM_AUTO_WB` | True | Set False to lock white balance |

## Project Structure

```
app.py              — Desktop inference app (OpenCV)
web_app.py          — Flask web dashboard
model_utils.py      — Shared model loading, inference, camera, anti-flicker
config.py           — All tunable settings
session_logger.py   — Session event logging
report_generator.py — PDF + CSV report generation
convert.py          — H5 → TFLite converter utility
templates/          — Flask HTML templates
sessions/           — Session output directories
```
