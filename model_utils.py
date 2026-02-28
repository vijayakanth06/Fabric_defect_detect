"""
model_utils.py — Shared model loading, preprocessing, inference & localisation.
================================================================================
All other modules (app.py, web_app.py) import from here.
"""

import os
import threading
import time
from collections import deque

import cv2
import numpy as np

import config

# ─────────────────────────────────────────────────────────────────────────────
#  PROJECT ROOT  (so paths work from any cwd)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve(path: str) -> str:
    """Resolve a path relative to the project root if it isn't absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_tflite(model_path: str | None = None):
    """
    Load a TFLite model.  Returns (interpreter, input_details, output_details).
    """
    import tensorflow as tf

    path = _resolve(model_path or config.TFLITE_MODEL_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"TFLite model not found: {path}")

    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    outp = interpreter.get_output_details()
    print(f"[Model] Loaded TFLite  ← {path}")
    return interpreter, inp, outp


def load_h5(model_path: str | None = None):
    """
    Load a Keras H5 model.  Returns the compiled model object.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import InputLayer as KerasInputLayer
    from tensorflow.keras import mixed_precision

    class LegacyInputLayer(KerasInputLayer):
        @classmethod
        def from_config(cls, cfg):
            if "batch_shape" in cfg and "batch_input_shape" not in cfg:
                cfg["batch_input_shape"] = tuple(cfg.pop("batch_shape"))
            return super().from_config(cfg)

    path = _resolve(model_path or config.H5_MODEL_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"H5 model not found: {path}")

    custom = {
        "InputLayer": LegacyInputLayer,
        "DTypePolicy": mixed_precision.Policy,
    }
    model = tf.keras.models.load_model(path, custom_objects=custom, compile=False)
    print(f"[Model] Loaded H5      ← {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  PREPROCESSING  (cv2.resize — faster than PIL)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(frame: np.ndarray) -> np.ndarray:
    """BGR frame → float32 tensor (1, H, W, 3) normalised to [0,1]."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, config.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_batch(frames: list[np.ndarray]) -> np.ndarray:
    """Preprocess multiple frames into a single batch tensor (N, H, W, 3)."""
    batch = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, config.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        batch.append(resized.astype(np.float32) / 255.0)
    return np.array(batch)


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE  (with optional TTA)
# ─────────────────────────────────────────────────────────────────────────────
def _make_augments(frame: np.ndarray, n: int) -> list[np.ndarray]:
    """Return up to `n` augmented views of frame (original first)."""
    augs = [frame]
    if n >= 2:
        augs.append(cv2.flip(frame, 1))  # horizontal flip
    if n >= 3:
        augs.append(cv2.flip(frame, 0))  # vertical flip
    return augs[:n]


def predict_tflite(frame: np.ndarray, interpreter, inp_details, outp_details,
                   tta_passes: int | None = None) -> tuple[str, float]:
    """Run TFLite inference with optional TTA. Returns (class_name, confidence)."""
    passes = tta_passes if tta_passes is not None else config.TTA_PASSES
    preds = []
    for aug in _make_augments(frame, passes):
        t = preprocess(aug)
        interpreter.set_tensor(inp_details[0]["index"], t)
        interpreter.invoke()
        preds.append(interpreter.get_tensor(outp_details[0]["index"]))
    avg = np.mean(preds, axis=0)
    return config.CLASS_NAMES[int(np.argmax(avg))], float(np.max(avg))


def predict_h5(frame: np.ndarray, model,
               tta_passes: int | None = None) -> tuple[str, float]:
    """
    Run H5/Keras inference with optional TTA.
    Uses batch predict for efficiency when TTA > 1.
    """
    passes = tta_passes if tta_passes is not None else config.TTA_PASSES
    augments = _make_augments(frame, passes)

    if len(augments) == 1:
        pred = model.predict(preprocess(augments[0]), verbose=0)
    else:
        # Batch all augments into a single predict call — much faster
        batch = preprocess_batch(augments)
        pred = model.predict(batch, verbose=0)

    avg = np.mean(pred, axis=0) if pred.ndim > 1 and pred.shape[0] > 1 else pred[0]
    return config.CLASS_NAMES[int(np.argmax(avg))], float(np.max(avg))


def make_predict_fn(model_type: str, **kwargs):
    """
    Factory: load model and return a predict_fn(frame) → (class, confidence).
    """
    if model_type == "tflite":
        interp, inp_d, outp_d = load_tflite(kwargs.get("model_path"))
        return lambda f: predict_tflite(f, interp, inp_d, outp_d)
    else:
        model = load_h5(kwargs.get("model_path"))
        return lambda f: predict_h5(f, model)


# ─────────────────────────────────────────────────────────────────────────────
#  SPATIAL LOCALISATION  (optional — off by default)
# ─────────────────────────────────────────────────────────────────────────────
def localise_defect(frame: np.ndarray, predict_fn,
                    grid: tuple[int, int] = (3, 3)) -> tuple[tuple, float]:
    """
    Divide frame into grid patches, predict each (with TTA=1 for speed).
    Returns (bbox, confidence) of highest-confidence defect patch.
    Falls back to full-frame if no patch beats threshold.
    """
    H, W = frame.shape[:2]
    rows, cols = grid
    ph, pw = H // rows, W // cols

    best_conf = -1.0
    best_bbox = (0, 0, W, H)

    for r in range(rows):
        for c in range(cols):
            y1, x1 = r * ph, c * pw
            patch = frame[y1: y1 + ph, x1: x1 + pw]
            cls, conf = predict_fn(patch)
            if cls != config.DEFECT_FREE_CLASS and conf > best_conf:
                best_conf = conf
                best_bbox = (x1, y1, pw, ph)

    return best_bbox, best_conf


# ─────────────────────────────────────────────────────────────────────────────
#  FAST OPENCV-BASED BOUNDING BOX  (sub-millisecond, no model calls)
# ─────────────────────────────────────────────────────────────────────────────
def _centered_fallback(W: int, H: int) -> tuple:
    """Return a centered box covering ~40 % of the frame as a fallback."""
    margin_x = int(W * 0.20)
    margin_y = int(H * 0.20)
    return (margin_x, margin_y, W - 2 * margin_x, H - 2 * margin_y)


def fast_localise_defect(frame: np.ndarray, padding: int = 24) -> tuple:
    """
    Use pure OpenCV image processing to locate the defect region.
    Runs in <1 ms — zero FPS impact compared to grid-based localisation.

    Uses three complementary methods and merges results:
      1. Gaussian-blur difference (texture anomalies)
      2. Laplacian magnitude (edges / scratches / lines)
      3. Adaptive threshold (local intensity outliers)

    Returns (x, y, w, h) bounding box.  If nothing specific is found,
    returns a centered ~40 % box so a visible bbox always appears.
    """
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Method 1: Gaussian-blur difference ────────────────────────
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = cv2.absdiff(gray, blurred)
    _, t1 = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    # ── Method 2: Laplacian edges ─────────────────────────────────
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    lap_abs = cv2.convertScaleAbs(lap)
    _, t2 = cv2.threshold(lap_abs, 40, 255, cv2.THRESH_BINARY)

    # ── Method 3: Adaptive threshold (local outliers) ─────────────
    t3 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 8,
    )

    # ── Combine all three masks ───────────────────────────────────
    combined = cv2.bitwise_or(t1, cv2.bitwise_or(t2, t3))

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _centered_fallback(W, H)

    # Keep contours that are at least 0.1 % of the frame area
    min_area = H * W * 0.001
    significant = [c for c in contours if cv2.contourArea(c) > min_area]

    if not significant:
        # Use the single largest contour
        significant = [max(contours, key=cv2.contourArea)]

    # Bounding rect encompassing all significant contours
    all_pts = np.concatenate(significant)
    x, y, w, h = cv2.boundingRect(all_pts)

    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(W - x, w + 2 * padding)
    h = min(H - y, h + 2 * padding)

    # If bbox still covers >80 % of frame, return centered fallback
    if w * h > 0.80 * W * H:
        return _centered_fallback(W, H)

    return (x, y, w, h)


# ─────────────────────────────────────────────────────────────────────────────
#  EMA PREDICTION SMOOTHER  (anti-flicker)
# ─────────────────────────────────────────────────────────────────────────────
class PredictionSmoother:
    """
    Exponential moving average over prediction logits to reduce
    class flickering caused by camera auto-exposure / noise.
    """

    def __init__(self, n_classes: int = len(config.CLASS_NAMES),
                 alpha: float | None = None):
        self.alpha = alpha if alpha is not None else config.EMA_ALPHA
        self.ema = np.zeros(n_classes, dtype=np.float64)
        self._initialized = False

    def smooth(self, pred_class: str, confidence: float) -> tuple[str, float]:
        """
        Feed raw prediction → get smoothed prediction.
        """
        if not config.ENABLE_EMA_SMOOTHING:
            return pred_class, confidence

        # Build a one-hot-ish vector weighted by confidence
        idx = config.CLASS_NAMES.index(pred_class)
        vec = np.zeros(len(config.CLASS_NAMES), dtype=np.float64)
        vec[idx] = confidence

        if not self._initialized:
            self.ema = vec.copy()
            self._initialized = True
        else:
            self.ema = self.alpha * vec + (1 - self.alpha) * self.ema

        smoothed_idx = int(np.argmax(self.ema))
        smoothed_conf = float(self.ema[smoothed_idx])
        return config.CLASS_NAMES[smoothed_idx], smoothed_conf

    def reset(self):
        self.ema[:] = 0
        self._initialized = False


# ─────────────────────────────────────────────────────────────────────────────
#  THREADED CAMERA CAPTURE  (decouples capture FPS from inference FPS)
# ─────────────────────────────────────────────────────────────────────────────
class CameraStream:
    """
    Threaded camera reader.  Always holds the *latest* frame;
    inference loop grabs it when ready — no frame queue build-up.
    """

    def __init__(self, src: int | str = 0, resolution: tuple[int, int] = (1280, 720)):
        self.cap = cv2.VideoCapture(src)

        # Resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # Buffer size = 1  → always read the newest frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Apply camera anti-flicker settings from config
        self._apply_camera_settings()

        self._lock = threading.Lock()
        self._frame = None
        self._stopped = False
        self._thread = threading.Thread(target=self._reader, daemon=True)

    def _apply_camera_settings(self):
        """Apply camera properties from config to reduce flickering."""
        if not config.CAM_AUTO_EXPOSURE:
            # 0.25 = manual mode on many cameras; 1 = manual on DirectShow
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            if config.CAM_EXPOSURE != 0:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, config.CAM_EXPOSURE)
        if config.CAM_GAIN >= 0:
            self.cap.set(cv2.CAP_PROP_GAIN, config.CAM_GAIN)
        if not config.CAM_AUTO_WB:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            if config.CAM_WB_TEMPERATURE > 0:
                self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, config.CAM_WB_TEMPERATURE)

    def start(self) -> "CameraStream":
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera.")
        ret, frame = self.cap.read()
        if ret:
            self._frame = frame
        self._thread.start()
        return self

    def _reader(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            if not ret:
                self._stopped = True
                break
            with self._lock:
                self._frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    @property
    def is_opened(self) -> bool:
        return self.cap.isOpened() and not self._stopped

    def release(self):
        self._stopped = True
        self._thread.join(timeout=2)
        self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
#  CLAHE PREPROCESSING  (optional — helps normalise uneven lighting)
# ─────────────────────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel in LAB space for brightness normalisation."""
    if not config.ENABLE_CLAHE:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
