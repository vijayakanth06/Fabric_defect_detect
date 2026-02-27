"""
config.py — Central configuration for Fabric Defect Detector v2
All tunable constants live here. Edit this file to change behaviour.
"""

import os

# ─────────────────────────────────────────────
# PROJECT ROOT  (absolute path — all relative paths resolve from here)
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
MODEL_TYPE        = "tflite"                                         # "tflite" | "h5"
TFLITE_MODEL_PATH = os.path.join(PROJECT_ROOT, "fabric_defect_model_quant.tflite")
H5_MODEL_PATH     = os.path.join(PROJECT_ROOT, "fabric_defect_model.h5")

# ─────────────────────────────────────────────
# CLASSES  (must match training order)
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "broken_stitch",
    "defect-free",
    "hole",
    "horizontal",
    "lines",
    "needle_mark",
    "pinched_fabric",
    "stain",
    "vertical",
]

DEFECT_FREE_CLASS = "defect-free"

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
INPUT_SIZE        = (224, 224)          # (width, height) fed to model
DEFECT_THRESHOLD  = 0.85               # Min confidence to flag as defect
TTA_PASSES        = 1                  # Test-Time Augmentation passes (1=fastest, 3=most accurate)

# ─────────────────────────────────────────────
# PERFORMANCE
# ─────────────────────────────────────────────
INFERENCE_SKIP_FRAMES = 2              # run inference every N frames (1=every frame, 2=every other)
ENABLE_LOCALISATION   = False          # enable 3×3 grid defect localisation (slower)
LOCALISATION_GRID     = (3, 3)         # grid rows × cols for patch-based localisation

# ─────────────────────────────────────────────
# CAMERA  (anti-flicker settings)
# ─────────────────────────────────────────────
# Set CAM_AUTO_EXPOSURE = False to lock exposure and prevent flickering.
# Tune CAM_EXPOSURE for your lighting setup (typical range: -10 to 0).
CAM_AUTO_EXPOSURE   = True             # True = camera auto-adjusts ; False = manual
CAM_EXPOSURE        = -6               # manual exposure value (used when auto=False)
CAM_GAIN            = -1               # -1 = don't touch; ≥0 = manual gain
CAM_AUTO_WB         = True             # True = auto white balance
CAM_WB_TEMPERATURE  = 4500             # manual WB temperature (used when auto_wb=False)
CAM_RESOLUTION      = (1280, 720)      # webcam resolution (width, height)

# ─────────────────────────────────────────────
# PREDICTION SMOOTHING  (anti-jitter)
# ─────────────────────────────────────────────
ENABLE_EMA_SMOOTHING = True            # smooth predictions across frames
EMA_ALPHA            = 0.5             # 0→very smooth (laggy); 1→raw (no smoothing)

# ─────────────────────────────────────────────
# CLAHE PREPROCESSING  (optional — normalises uneven lighting)
# ─────────────────────────────────────────────
ENABLE_CLAHE = False                   # enable CLAHE brightness normalisation

# ─────────────────────────────────────────────
# INPUT SOURCE
# ─────────────────────────────────────────────
SOURCE       = "webcam"                # "webcam" | "image" | "video"
SOURCE_PATH  = ""                      # path to image/video (when SOURCE != "webcam")
WEBCAM_INDEX = 0                       # camera device index

# ─────────────────────────────────────────────
# SESSION & REPORTING
# ─────────────────────────────────────────────
SESSIONS_DIR          = os.path.join(PROJECT_ROOT, "sessions")
SAVE_DEFECT_FRAMES    = True           # save frame of each detected defect
MIN_REPORT_CONFIDENCE = 0.70           # include events above this in report

# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────
WINDOW_TITLE   = "Fabric Defect Detector v2"
FONT_SCALE     = 0.75
TEXT_THICKNESS  = 2
COLOR_OK       = (0, 220, 0)           # BGR green  — defect-free
COLOR_DEFECT   = (0, 0, 220)           # BGR red    — defect
COLOR_INFO     = (220, 220, 220)       # BGR white  — HUD text
BOX_THICKNESS  = 8                     # border around frame on defect
