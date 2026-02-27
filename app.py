"""
app.py — Fabric Defect Detector v2
====================================
Unified inference script with threaded capture and anti-flicker.

Usage
-----
  python app.py                              # live webcam (TFLite, default)
  python app.py --model h5                   # live webcam (H5/Keras)
  python app.py --source image --path img.jpg
  python app.py --source video --path vid.mp4

Keyboard Shortcuts (live window)
---------------------------------
  q  : quit + generate report
  r  : generate report NOW (session continues)
  s  : manually save current frame snapshot
  p  : pause / resume
"""

import argparse
import os
import sys
import time
import traceback

import cv2
import numpy as np

# ── local modules ─────────────────────────────────────────────────────────────
import config
from model_utils import (
    make_predict_fn,
    localise_defect,
    PredictionSmoother,
    CameraStream,
    apply_clahe,
)
from session_logger import SessionLogger


# ─────────────────────────────────────────────────────────────────────────────
# HUD OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put_text(frame, text, org, scale=0.65, color=(220, 220, 220), thickness=2):
    cv2.putText(frame, text, (org[0] + 1, org[1] + 1), FONT, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_hud(frame, pred_class, confidence, fps, session: SessionLogger, paused):
    H, W = frame.shape[:2]
    is_defect = pred_class != config.DEFECT_FREE_CLASS and confidence >= config.DEFECT_THRESHOLD

    if is_defect:
        cv2.rectangle(frame, (0, 0), (W - 1, H - 1),
                      config.COLOR_DEFECT, config.BOX_THICKNESS)

    # ── top-left panel ────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 110), (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    color = config.COLOR_DEFECT if is_defect else config.COLOR_OK
    label = pred_class.upper().replace("_", " ")
    _put_text(frame, label,                                    (10, 28),  scale=0.85, color=color, thickness=2)
    _put_text(frame, f"Confidence : {confidence:.1%}",         (10, 55),  scale=0.60)
    _put_text(frame, f"FPS        : {fps:.1f}",                (10, 78),  scale=0.60)
    _put_text(frame, f"Defects    : {session.defect_count}",   (10, 101), scale=0.60)

    # ── bottom-right session timer ────────────────────────────────
    elapsed   = int(session.session_duration)
    dur_label = f"Session {elapsed // 60:02d}:{elapsed % 60:02d}"
    (tw, _), _ = cv2.getTextSize(dur_label, FONT, 0.55, 2)
    _put_text(frame, dur_label, (W - tw - 14, H - 12), scale=0.55)

    if paused:
        _put_text(frame, "[ PAUSED ]", (W // 2 - 70, H // 2),
                  scale=1.2, color=(0, 220, 255), thickness=3)

    if is_defect:
        (tw, _), _ = cv2.getTextSize("DEFECT DETECTED", FONT, 1.2, 3)
        _put_text(frame, "DEFECT DETECTED", (W // 2 - tw // 2, 120),
                  scale=1.2, color=(0, 60, 255), thickness=3)

    _put_text(frame, "q:quit  r:report  s:snap  p:pause",
              (10, H - 12), scale=0.45, color=(160, 160, 160), thickness=1)


def draw_bbox(frame, bbox, label, confidence):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), config.COLOR_DEFECT, 2)
    tag = f"{label} {confidence:.0%}"
    (tw, th), bl = cv2.getTextSize(tag, FONT, 0.55, 2)
    cv2.rectangle(frame, (x, y - th - bl - 4), (x + tw + 4, y),
                  config.COLOR_DEFECT, -1)
    cv2.putText(frame, tag, (x + 2, y - bl - 2), FONT, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run(model_type: str, source: str, source_path: str, webcam_index: int):
    # ── load model ────────────────────────────────────────────────
    predict_fn = make_predict_fn(model_type)

    # ── open source ───────────────────────────────────────────────
    use_threaded = source == "webcam"
    cam_stream = None

    if use_threaded:
        cam_stream = CameraStream(
            src=webcam_index,
            resolution=config.CAM_RESOLUTION,
        ).start()
    else:
        if source in ("video", "image"):
            if not source_path or not os.path.exists(source_path):
                raise FileNotFoundError(f"Source file not found: {source_path}")
        cap = cv2.VideoCapture(source_path if source != "webcam" else webcam_index)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video source.")

    # ── session ───────────────────────────────────────────────────
    session = SessionLogger()
    session.start()

    smoother = PredictionSmoother()

    print(f"\n{'=' * 55}")
    print("  Fabric Defect Detector v2")
    print(f"  Model : {model_type.upper()}")
    print(f"  Source: {source}  ({source_path or webcam_index})")
    print(f"  TTA={config.TTA_PASSES}  Skip={config.INFERENCE_SKIP_FRAMES}"
          f"  Localise={config.ENABLE_LOCALISATION}  EMA={config.ENABLE_EMA_SMOOTHING}")
    print(f"  Session dir: {session.session_dir}")
    print(f"{'=' * 55}")
    print("  Shortcuts:  q=quit  r=report  s=snapshot  p=pause")
    print(f"{'=' * 55}\n")

    paused           = False
    fps_smooth       = 30.0
    frame_saved_flag = False
    frame_counter    = 0

    # Cached prediction for frames we skip
    cached_class = config.DEFECT_FREE_CLASS
    cached_conf  = 0.0
    cached_bbox  = (0, 0, 640, 480)

    while True:
        # ── read frame ────────────────────────────────────────────
        if use_threaded:
            ret, frame = cam_stream.read()
            if not cam_stream.is_opened:
                print("[App] Camera disconnected.")
                break
        else:
            ret, frame = cap.read()

        if source == "image":
            if not ret:
                break
        elif not ret:
            print("[App] End of source.")
            break

        if paused:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("p"):
                paused = False
            elif key == ord("q"):
                break
            cv2.imshow(config.WINDOW_TITLE, frame)
            continue

        session.tick()
        frame_counter += 1
        t0 = time.time()

        # ── optional CLAHE preprocessing ──────────────────────────
        processed = apply_clahe(frame)

        # ── inference (with frame skipping) ───────────────────────
        run_inference = (frame_counter % config.INFERENCE_SKIP_FRAMES == 0)

        if run_inference:
            try:
                raw_class, raw_conf = predict_fn(processed)
                # EMA smoothing to reduce jitter
                pred_class, confidence = smoother.smooth(raw_class, raw_conf)
            except Exception as e:
                print(f"[Inference Error] {e}")
                pred_class, confidence = config.DEFECT_FREE_CLASS, 0.0

            is_defect = (pred_class != config.DEFECT_FREE_CLASS
                         and confidence >= config.DEFECT_THRESHOLD)
            bbox = (0, 0, frame.shape[1], frame.shape[0])

            # ── optional localisation ─────────────────────────────
            if is_defect and config.ENABLE_LOCALISATION:
                bbox, _ = localise_defect(processed, predict_fn,
                                          grid=config.LOCALISATION_GRID)

            cached_class = pred_class
            cached_conf  = confidence
            cached_bbox  = bbox
        else:
            pred_class = cached_class
            confidence = cached_conf
            bbox       = cached_bbox
            is_defect  = (pred_class != config.DEFECT_FREE_CLASS
                          and confidence >= config.DEFECT_THRESHOLD)

        # ── draw bbox if defect ───────────────────────────────────
        if is_defect:
            draw_bbox(frame, bbox, pred_class, confidence)
            if not frame_saved_flag:
                session.log_event(frame.copy(), pred_class, confidence, bbox)
                frame_saved_flag = True
        else:
            frame_saved_flag = False

        # ── FPS (measures display loop, not just inference) ───────
        elapsed = time.time() - t0
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(elapsed, 1e-6))

        # ── HUD ───────────────────────────────────────────────────
        draw_hud(frame, pred_class, confidence, fps_smooth, session, paused)
        cv2.imshow(config.WINDOW_TITLE, frame)

        # ── keyboard ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            _generate_report(session)
        elif key == ord("s"):
            snap_dir  = os.path.join(session.session_dir, "frames")
            snap_name = f"manual_snap_{session.total_frames:06d}.jpg"
            snap_path = os.path.join(snap_dir, snap_name)
            cv2.imwrite(snap_path, frame)
            print(f"[Snap] Saved → {snap_path}")
        elif key == ord("p"):
            paused = True

        if source == "image":
            cv2.waitKey(0)
            break

    # ── teardown ──────────────────────────────────────────────────
    if use_threaded and cam_stream:
        cam_stream.release()
    elif not use_threaded:
        cap.release()

    cv2.destroyAllWindows()
    session.stop()
    _generate_report(session)


def _generate_report(session: SessionLogger):
    try:
        from report_generator import generate_pdf_report
        session._save_json()
        json_path = os.path.join(session.session_dir, "session_log.json")
        pdf = generate_pdf_report(json_path, session.session_dir)
        print(f"[Report] Generated → {pdf}")
    except Exception as e:
        print(f"[Report Error] {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Fabric Defect Detector v2")
    p.add_argument("--model",   default=config.MODEL_TYPE,
                   choices=["tflite", "h5"], help="Model backend")
    p.add_argument("--source",  default=config.SOURCE,
                   choices=["webcam", "image", "video"], help="Input source")
    p.add_argument("--path",    default=config.SOURCE_PATH,
                   help="Path to image/video file (when --source != webcam)")
    p.add_argument("--camera",  default=config.WEBCAM_INDEX, type=int,
                   help="Webcam device index")
    p.add_argument("--tta",     default=None, type=int, choices=[1, 2, 3],
                   help="Override TTA passes (1=fastest, 3=most accurate)")
    p.add_argument("--skip",    default=None, type=int,
                   help="Override inference skip frames (1=every, 3=every 3rd)")
    p.add_argument("--localise", action="store_true", default=None,
                   help="Enable defect localisation (slower)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Apply CLI overrides to config
    if args.tta is not None:
        config.TTA_PASSES = args.tta
    if args.skip is not None:
        config.INFERENCE_SKIP_FRAMES = args.skip
    if args.localise is not None:
        config.ENABLE_LOCALISATION = args.localise

    try:
        run(
            model_type   = args.model,
            source       = args.source,
            source_path  = args.path,
            webcam_index = args.camera,
        )
    except KeyboardInterrupt:
        print("\n[App] Interrupted by user.")
    except Exception as e:
        print(f"\n[Fatal] {e}")
        traceback.print_exc()
        sys.exit(1)
