"""
session_logger.py — Records every defect event during a live session.

Each DetectionEvent stores:
  - frame_number      : int
  - timestamp         : ISO-8601 string (wall-clock time)
  - elapsed_sec       : float (seconds since session start)
  - predicted_class   : str
  - confidence        : float  (0–1)
  - bbox              : (x, y, w, h) in pixels — region of interest
  - frame_path        : str | None  — path to saved frame crop
"""

import os
import json
import time
import cv2
import numpy as np
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Tuple, List

from config import SESSIONS_DIR, SAVE_DEFECT_FRAMES, DEFECT_FREE_CLASS


@dataclass
class DetectionEvent:
    frame_number:    int
    timestamp:       str               # ISO-8601
    elapsed_sec:     float
    predicted_class: str
    confidence:      float
    bbox:            Tuple[int, int, int, int]   # x, y, w, h
    frame_path:      Optional[str] = None


class SessionLogger:
    """
    Create one per inference session.  Call .start() before loop,
    .log_event() for every detected defect, .stop() to finalise.
    """

    def __init__(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id   = ts
        self.session_dir  = os.path.join(SESSIONS_DIR, f"session_{ts}")
        self.frames_dir   = os.path.join(self.session_dir, "frames")

        os.makedirs(self.frames_dir, exist_ok=True)

        self.start_time:   Optional[float] = None
        self.end_time:     Optional[float] = None
        self.total_frames: int = 0
        self.events:       List[DetectionEvent] = []

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self):
        self.start_time = time.time()
        print(f"[SessionLogger] Session started  → {self.session_dir}")

    def stop(self):
        self.end_time = time.time()
        self._save_json()
        print(f"[SessionLogger] Session ended. {len(self.events)} defect(s) logged.")

    # ── recording ─────────────────────────────────────────────────────────
    def tick(self):
        """Call once per captured frame to increment frame counter."""
        self.total_frames += 1

    def log_event(
        self,
        frame:           np.ndarray,
        predicted_class: str,
        confidence:      float,
        bbox:            Tuple[int, int, int, int],
    ) -> DetectionEvent:
        """
        Record a defect detection.

        Parameters
        ----------
        frame           : current BGR frame from OpenCV
        predicted_class : one of CLASS_NAMES
        confidence      : model softmax confidence (0–1)
        bbox            : (x, y, w, h) — bounding box in pixel coords
        """
        now          = datetime.now()
        elapsed      = time.time() - self.start_time
        frame_path   = None

        if SAVE_DEFECT_FRAMES and predicted_class != DEFECT_FREE_CLASS:
            fname      = f"frame_{self.total_frames:06d}_{predicted_class}_{confidence:.2f}.jpg"
            frame_path = os.path.join(self.frames_dir, fname)
            # Save the annotated frame (caller should draw bbox before passing)
            cv2.imwrite(frame_path, frame)

        event = DetectionEvent(
            frame_number    = self.total_frames,
            timestamp       = now.isoformat(timespec="seconds"),
            elapsed_sec     = round(elapsed, 2),
            predicted_class = predicted_class,
            confidence      = round(float(confidence), 4),
            bbox            = bbox,
            frame_path      = frame_path,
        )
        self.events.append(event)
        return event

    # ── persistence ────────────────────────────────────────────────────────
    def _save_json(self):
        # end_time may be None when report is requested mid-session
        end_ts   = self.end_time if self.end_time is not None else time.time()
        duration = round(end_ts - self.start_time, 2)
        payload = {
            "session_id":   self.session_id,
            "start_time":   datetime.fromtimestamp(self.start_time).isoformat(timespec="seconds"),
            "end_time":     datetime.fromtimestamp(end_ts).isoformat(timespec="seconds"),
            "duration_sec": duration,
            "total_frames": self.total_frames,
            "defect_count": len(self.events),
            "events":       [asdict(e) for e in self.events],
        }
        path = os.path.join(self.session_dir, "session_log.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[SessionLogger] JSON log saved → {path}")
        return path

    # ── helpers ────────────────────────────────────────────────────────────
    @property
    def defect_count(self) -> int:
        return len(self.events)

    @property
    def session_duration(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def class_counts(self) -> dict:
        counts = {}
        for e in self.events:
            counts[e.predicted_class] = counts.get(e.predicted_class, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
