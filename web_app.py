"""
web_app.py  --  Fabric Defect Detector v2  |  Flask Live Web Dashboard
=======================================================================
Real-time MJPEG webcam stream with per-frame inference.

Run:
    python web_app.py
    python web_app.py --model h5
    python web_app.py --camera 1

Then open http://localhost:5000 in your browser.
"""

import os
import sys
import json
import time
import threading
import argparse

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, send_file, request

import config
from model_utils import (
    make_predict_fn,
    localise_defect,
    fast_localise_defect,
    PredictionSmoother,
    CameraStream,
    apply_clahe,
    PROJECT_ROOT,
)
from session_logger   import SessionLogger
from report_generator import generate_pdf_report

app = Flask(__name__, template_folder=os.path.join(PROJECT_ROOT, "templates"))

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBALS  (shared between stream thread and Flask routes)
# ─────────────────────────────────────────────────────────────────────────────
_lock = threading.Lock()

state = {
    "pred_class":   config.DEFECT_FREE_CLASS,
    "confidence":   0.0,
    "bbox":         [0, 0, 640, 480],
    "fps":          0.0,
    "is_defect":    False,
    "frame_count":  0,
    "defect_count": 0,
}

session_obj: SessionLogger = None
session_active: bool = False
events_list: list = []

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────────────────────
#  FRAME ANNOTATION
# ─────────────────────────────────────────────────────────────────────────────
def _put(img, txt, org, scale=0.65, color=(220, 220, 220), thick=2):
    cv2.putText(img, txt, (org[0]+1, org[1]+1), FONT, scale, (0,0,0), thick+1, cv2.LINE_AA)
    cv2.putText(img, txt, org,                  FONT, scale, color,   thick,   cv2.LINE_AA)


def annotate(frame, pred_class, confidence, bbox, fps):
    out = frame.copy()
    H, W = out.shape[:2]
    is_defect = (pred_class != config.DEFECT_FREE_CLASS
                 and confidence >= config.DEFECT_THRESHOLD)

    if is_defect:
        x, y, w, h = bbox

        # Draw the localised bounding box
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 255), 3)

        # Label above the box (or inside if near top edge)
        tag = f"{pred_class}  {confidence:.0%}"
        (tw, th), bl = cv2.getTextSize(tag, FONT, 0.6, 2)
        if y > th + bl + 6:
            label_y = y - bl - 2
            cv2.rectangle(out, (x, y-th-bl-4), (x+tw+6, y), (0,0,255), -1)
        else:
            label_y = y + th + bl + 6
            cv2.rectangle(out, (x, y+2), (x+tw+6, y+th+bl+10), (0,0,255), -1)
        cv2.putText(out, tag, (x+3, label_y), FONT, 0.6, (255,255,255), 2)

        banner = "DEFECT DETECTED"
        (bw, _), _ = cv2.getTextSize(banner, FONT, 1.1, 3)
        cv2.putText(out, banner, ((W-bw)//2+1, 111), FONT, 1.1, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(out, banner, ((W-bw)//2, 110),   FONT, 1.1, (0,60,255), 3, cv2.LINE_AA)

    # HUD panel
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (305, 108), (15, 15, 35), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    label = pred_class.upper().replace("_", " ")
    color = (0, 220, 0) if not is_defect else (0, 60, 255)
    _put(out, label,                      (10, 28), scale=0.85, color=color, thick=2)
    _put(out, f"Conf  : {confidence:.1%}",(10, 55), scale=0.60)
    _put(out, f"FPS   : {fps:.1f}",       (10, 78), scale=0.60)
    _put(out, "Session active" if session_active else "No session",
              (10, 101), scale=0.50, color=(200,200,80) if session_active else (120,120,120))

    hint = "[ START SESSION to log events ]" if not session_active else "[LOGGING DEFECTS]"
    _put(out, hint, (10, H-12), scale=0.42, color=(120,120,120), thick=1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  MJPEG GENERATOR  (runs in its own thread via Flask Response)
# ─────────────────────────────────────────────────────────────────────────────
_DEFECT_COOLDOWN = 1.5
_last_log_t = 0.0


def gen_frames(predict_fn, camera_index):
    global session_obj, session_active, events_list, _last_log_t

    cam = CameraStream(src=camera_index, resolution=config.CAM_RESOLUTION).start()
    smoother = PredictionSmoother()

    fps_smooth    = 30.0
    frame_counter = 0
    cached_class  = config.DEFECT_FREE_CLASS
    cached_conf   = 0.0
    cached_bbox   = [0, 0, 640, 480]

    while cam.is_opened:
        ret, frame = cam.read()
        if not ret:
            continue

        frame_counter += 1
        t0 = time.time()

        # Optional CLAHE
        processed = apply_clahe(frame)

        # Frame skipping — only run inference every N frames
        run_inference = (frame_counter % config.INFERENCE_SKIP_FRAMES == 0)

        if run_inference:
            try:
                raw_class, raw_conf = predict_fn(processed)
                pred_class, confidence = smoother.smooth(raw_class, raw_conf)
            except Exception:
                pred_class, confidence = config.DEFECT_FREE_CLASS, 0.0

            is_defect = (pred_class != config.DEFECT_FREE_CLASS
                         and confidence >= config.DEFECT_THRESHOLD)

            bbox = [0, 0, frame.shape[1], frame.shape[0]]
            if is_defect and config.ENABLE_LOCALISATION:
                bbox_t, _ = localise_defect(processed, predict_fn,
                                            grid=config.LOCALISATION_GRID)
                bbox = list(bbox_t)
            elif is_defect and config.ENABLE_FAST_BBOX:
                bbox = list(fast_localise_defect(processed))

            cached_class = pred_class
            cached_conf  = confidence
            cached_bbox  = bbox
        else:
            pred_class = cached_class
            confidence = cached_conf
            bbox       = cached_bbox
            is_defect  = (pred_class != config.DEFECT_FREE_CLASS
                          and confidence >= config.DEFECT_THRESHOLD)

        # FPS
        elapsed = time.time() - t0
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(elapsed, 1e-6))

        # Update shared state
        with _lock:
            state["pred_class"]   = pred_class
            state["confidence"]   = round(confidence, 4)
            state["bbox"]         = bbox
            state["fps"]          = round(fps_smooth, 1)
            state["is_defect"]    = is_defect
            state["frame_count"] += 1
            if is_defect:
                state["defect_count"] += 1

        # Session logging (with cooldown)
        now = time.time()
        if is_defect and session_active and session_obj and (now - _last_log_t > _DEFECT_COOLDOWN):
            _last_log_t = now
            session_obj.tick()
            event = session_obj.log_event(frame.copy(), pred_class, confidence, bbox)
            events_list.append({
                "num":     len(events_list) + 1,
                "time":    event.timestamp,
                "elapsed": f"{event.elapsed_sec:.1f}s",
                "class":   pred_class,
                "conf":    f"{confidence:.1%}",
                "bbox":    f"({bbox[0]},{bbox[1]}) {bbox[2]}x{bbox[3]}",
            })
        elif session_active and session_obj:
            session_obj.tick()

        # Annotate
        out = annotate(frame, pred_class, confidence, bbox, fps_smooth)

        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n"
               + buf.tobytes()
               + b"\r\n")

    cam.release()


# ─────────────────────────────────────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────
_predict_fn   = None
_camera_index = 0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(_predict_fn, _camera_index),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/state")
def api_state():
    with _lock:
        s = dict(state)
    s["session_active"] = session_active
    s["defect_rate"] = f"{100 * s['defect_count'] / max(s['frame_count'], 1):.1f}%"
    return jsonify(s)


@app.route("/api/events")
def api_events():
    return jsonify(events_list)


@app.route("/api/session/start", methods=["POST"])
def session_start():
    global session_obj, session_active, events_list
    with _lock:
        state["frame_count"]  = 0
        state["defect_count"] = 0
    events_list   = []
    session_obj   = SessionLogger()
    session_obj.start()
    session_active = True
    return jsonify({"ok": True, "session_id": session_obj.session_id})


@app.route("/api/session/stop", methods=["POST"])
def session_stop():
    global session_obj, session_active
    if not session_obj:
        return jsonify({"ok": False, "error": "No active session"})
    session_obj.stop()
    session_active = False
    try:
        json_path = os.path.join(session_obj.session_dir, "session_log.json")
        pdf_path  = generate_pdf_report(json_path, session_obj.session_dir)
        return jsonify({
            "ok": True,
            "pdf": pdf_path,
            "csv": pdf_path.replace(".pdf", ".csv"),
            "session_id": session_obj.session_id,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/report/download/<session_id>/<fmt>")
def download_report(session_id, fmt):
    base = os.path.join(config.SESSIONS_DIR, f"session_{session_id}")
    if fmt == "pdf":
        path = os.path.join(base, f"report_{session_id}.pdf")
        mime = "application/pdf"
    else:
        path = os.path.join(base, f"report_{session_id}.csv")
        mime = "text/csv"
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(os.path.abspath(path), mimetype=mime,
                     as_attachment=True,
                     download_name=os.path.basename(path))


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fabric Defect Detector — Web App")
    parser.add_argument("--model",  default=config.MODEL_TYPE,
                        choices=["tflite", "h5"])
    parser.add_argument("--camera", default=config.WEBCAM_INDEX, type=int)
    parser.add_argument("--port",   default=5000, type=int)
    args = parser.parse_args()

    _camera_index = args.camera

    print(f"\n{'='*55}")
    print(f"  Fabric Defect Detector v2  |  Web App")
    print(f"  Loading model: {args.model.upper()} ...")
    print(f"{'='*55}")

    _predict_fn = make_predict_fn(args.model)

    print(f"  Model loaded. Open http://localhost:{args.port}")
    print(f"{'='*55}\n")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
