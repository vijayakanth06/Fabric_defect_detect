"""
report_generator.py -- Generates a professional PDF + CSV report for a session.

Dependencies: fpdf2, matplotlib
Install:  pip install fpdf2 matplotlib
"""

import os
import csv
import json
from datetime import datetime
from typing import List, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from fpdf import FPDF, HTMLMixin
from fpdf.enums import XPos, YPos

from config import CLASS_NAMES, DEFECT_FREE_CLASS, MIN_REPORT_CONFIDENCE


# -----------------------------------------------------------------------
#  COLOUR PALETTE (for charts)
# -----------------------------------------------------------------------
PALETTE = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#34495E", "#FF6B6B",
]

# Path to a bundled TTF font — we ship DejaVuSans which supports all latin chars,
# degrees, x-mark, arrows etc. fpdf2 includes it.
# We use the fpdf2 built-in "helvetica" for latin-1 text only and a unicode-safe
# font for any cells that may contain special chars.
_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")


def _get_unicode_font(pdf: FPDF) -> str:
    """
    Register and return a Unicode-capable font name.
    Tries to find DejaVuSans bundled with fpdf2 first, then falls back
    to Helvetica (latin-1 only — safe as long as text is ASCII-clean).
    """
    try:
        import fpdf
        pkg_dir   = os.path.dirname(fpdf.__file__)
        candidates = [
            os.path.join(pkg_dir, "fonts", "DejaVuSans.ttf"),
            os.path.join(pkg_dir, "fonts", "DejaVuSans.pkl"),
        ]
        for c in candidates:
            if os.path.exists(c):
                # PDF object-level font name
                pdf.add_font("DejaVu", fname=c)
                return "DejaVu"
    except Exception:
        pass
    return "Helvetica"


# -----------------------------------------------------------------------
#  CSV EXPORT
# -----------------------------------------------------------------------
def export_csv(events: List[Dict], out_path: str) -> str:
    fieldnames = [
        "frame_number", "timestamp", "elapsed_sec",
        "predicted_class", "confidence",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "frame_path",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in events:
            bbox = e.get("bbox", [0, 0, 0, 0])
            writer.writerow({
                "frame_number":    e["frame_number"],
                "timestamp":       e["timestamp"],
                "elapsed_sec":     e["elapsed_sec"],
                "predicted_class": e["predicted_class"],
                "confidence":      e["confidence"],
                "bbox_x":          bbox[0],
                "bbox_y":          bbox[1],
                "bbox_w":          bbox[2],
                "bbox_h":          bbox[3],
                "frame_path":      e.get("frame_path", ""),
            })
    return out_path


# -----------------------------------------------------------------------
#  CHART HELPERS
# -----------------------------------------------------------------------
def _bar_chart(counts: Dict[str, int], out_path: str):
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=9, color="#333333")
    ax.set_xlabel("Detections", fontsize=10)
    ax.set_title("Defect Class Distribution", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _timeline_chart(events: List[Dict], duration_sec: float, out_path: str):
    if not events:
        return

    defect_classes = [e["predicted_class"] for e in events
                      if e["predicted_class"] != DEFECT_FREE_CLASS]
    if not defect_classes:
        return

    unique_classes = sorted(set(defect_classes))
    color_map = {cls: PALETTE[i % len(PALETTE)] for i, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(10, 3))
    for e in events:
        if e["predicted_class"] == DEFECT_FREE_CLASS:
            continue
        x = e["elapsed_sec"]
        y = unique_classes.index(e["predicted_class"])
        ax.scatter(x, y, c=color_map[e["predicted_class"]], s=60, zorder=3)

    ax.set_yticks(range(len(unique_classes)))
    ax.set_yticklabels(unique_classes, fontsize=9)
    ax.set_xlabel("Elapsed Time (s)", fontsize=10)
    ax.set_title("Defect Timeline", fontsize=12, fontweight="bold")
    ax.set_xlim(-1, max(duration_sec + 1, 5))
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    patches = [mpatches.Patch(color=c, label=k) for k, c in color_map.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
#  PDF REPORT CLASS
# -----------------------------------------------------------------------
class _PDF(FPDF):
    """Custom FPDF subclass with themed header / footer (ASCII-safe text only)."""

    SESSION_ID  = ""
    UNICODE_FNT = "Helvetica"   # will be replaced with DejaVu if available

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(30, 30, 50)
        self.set_text_color(255, 255, 255)
        self.rect(0, 0, 210, 14, "F")
        self.set_xy(8, 3)
        # NOTE: only ASCII / latin-1 in core-font cells
        self.cell(0, 8, "Fabric Defect Detection - Session Report", align="L")
        self.set_xy(-70, 3)
        self.cell(60, 8, f"Session: {self.SESSION_ID}", align="R")
        self.set_text_color(0, 0, 0)
        self.ln(12)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 8, f"Page {self.page_no()} | Generated {now}", align="C")

    # ── helpers ────────────────────────────────────────────────────
    def section_title(self, title: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(240, 240, 248)
        self.set_text_color(30, 30, 80)
        self.cell(0, 8, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def kv_row(self, key: str, value: str, fill: bool = False):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(248, 248, 252)
        self.cell(55, 7, key, border=1, fill=fill)
        self.set_font("Helvetica", "", 9)
        # Ensure value is ASCII-safe for core fonts
        safe_val = value.encode("latin-1", errors="replace").decode("latin-1")
        self.cell(0, 7, safe_val, border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=False)

    def safe_cell(self, w, h, txt, **kwargs):
        """Render a cell replacing any non-latin-1 char with '?'."""
        safe = txt.encode("latin-1", errors="replace").decode("latin-1")
        self.cell(w, h, safe, **kwargs)


# -----------------------------------------------------------------------
#  MAIN REPORT FUNCTION
# -----------------------------------------------------------------------
def generate_pdf_report(session_json_path: str, out_dir: str) -> str:
    """
    Read a session_log.json produced by SessionLogger, write PDF + CSV.
    Returns path to created PDF.
    """
    with open(session_json_path, encoding="utf-8") as f:
        data = json.load(f)

    session_id   = data["session_id"]
    start_time   = data["start_time"]
    end_time     = data["end_time"]
    duration_sec = float(data["duration_sec"])
    total_frames = int(data["total_frames"])
    events: List[Dict] = data.get("events", [])

    # Filter to reportable events
    reported_events = [
        e for e in events
        if e["confidence"] >= MIN_REPORT_CONFIDENCE
        and e["predicted_class"] != DEFECT_FREE_CLASS
    ]

    defect_count = len(reported_events)
    fps_avg      = round(total_frames / max(duration_sec, 1), 1)
    defect_rate  = f"{100 * defect_count / max(total_frames, 1):.1f}%"

    class_counts: Dict[str, int] = {}
    for e in reported_events:
        class_counts[e["predicted_class"]] = class_counts.get(e["predicted_class"], 0) + 1

    # ── charts ────────────────────────────────────────────────────
    bar_chart_path      = os.path.join(out_dir, "_chart_bar.png")
    timeline_chart_path = os.path.join(out_dir, "_chart_timeline.png")

    if class_counts:
        _bar_chart(class_counts, bar_chart_path)
    _timeline_chart(reported_events, duration_sec, timeline_chart_path)

    # ── build PDF ─────────────────────────────────────────────────
    pdf = _PDF()
    pdf.SESSION_ID = session_id
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(12, 16, 12)
    pdf.add_page()

    # cover
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 30, 80)
    pdf.cell(0, 12, "Fabric Defect Inspection Report",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, "Automated Quality Control - Session Summary",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_draw_color(200, 200, 220)
    pdf.line(12, pdf.get_y(), 198, pdf.get_y())
    pdf.ln(6)
    pdf.set_text_color(0, 0, 0)

    # ── 1. Metadata ───────────────────────────────────────────────
    pdf.section_title("1. Session Metadata")
    dur_min = int(duration_sec // 60)
    dur_sec = int(duration_sec % 60)
    rows = [
        ("Session ID",        session_id),
        ("Start Time",        start_time),
        ("End Time",          end_time),
        ("Duration",          f"{duration_sec:.1f} s  ({dur_min}m {dur_sec}s)"),
        ("Total Frames",      str(total_frames)),
        ("Average FPS",       str(fps_avg)),
        ("Defects Detected",  str(defect_count)),
        ("Defect Rate",       defect_rate),
        ("Most Common Defect",
         max(class_counts, key=class_counts.get) if class_counts else "None"),
    ]
    for i, (k, v) in enumerate(rows):
        pdf.kv_row(k, v, fill=(i % 2 == 0))
    pdf.ln(4)

    # ── 2. Class distribution ─────────────────────────────────────
    if class_counts and os.path.exists(bar_chart_path):
        pdf.section_title("2. Class Distribution")
        pdf.image(bar_chart_path, x=12, w=130)
        pdf.ln(4)

    # ── 3. Timeline ───────────────────────────────────────────────
    if os.path.exists(timeline_chart_path) and reported_events:
        pdf.section_title("3. Defect Timeline")
        pdf.image(timeline_chart_path, x=12, w=180)
        pdf.ln(4)

    # ── 4. Event log ──────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("4. Defect Event Log")

    if not reported_events:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 8, "No defect events above confidence threshold.",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        col_w   = [12, 44, 18, 36, 14, 14, 14, 22]
        headers = ["#", "Timestamp", "Elapsed", "Defect Class",
                   "Conf", "BBox X", "BBox Y", "BBox WxH"]
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(30, 30, 80)
        pdf.set_text_color(255, 255, 255)
        for w, h in zip(col_w, headers):
            pdf.cell(w, 7, h, border=1, fill=True)
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

        for idx, e in enumerate(reported_events):
            pdf.set_font("Helvetica", "", 8)
            fill = idx % 2 == 0
            pdf.set_fill_color(248, 248, 252)

            bbox     = e.get("bbox", [0, 0, 0, 0])
            conf_str = f"{e['confidence']*100:.1f}%"
            bbox_wh  = f"{bbox[2]}x{bbox[3]}"   # ASCII 'x' not Unicode multiply

            # severity colour
            if e["confidence"] >= 0.90:
                pdf.set_text_color(180, 0, 0)
            elif e["confidence"] >= 0.75:
                pdf.set_text_color(160, 80, 0)
            else:
                pdf.set_text_color(0, 0, 0)

            row_vals = [
                str(idx + 1),
                e["timestamp"],
                f"{e['elapsed_sec']:.1f}s",
                e["predicted_class"],
                conf_str,
                str(bbox[0]),
                str(bbox[1]),
                bbox_wh,
            ]
            for w, v in zip(col_w, row_vals):
                # force ASCII-safe for core fonts
                safe_v = v.encode("latin-1", errors="replace").decode("latin-1")
                pdf.cell(w, 6, safe_v, border=1, fill=fill)
            pdf.ln()

        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 0, 0)
        pdf.cell(0, 5, "Red = confidence >= 90%   Orange = confidence >= 75%",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)

    # ── 5. Snapshots ──────────────────────────────────────────────
    snap_events = [
        e for e in reported_events
        if e.get("frame_path") and os.path.exists(e["frame_path"])
    ]
    if snap_events:
        pdf.add_page()
        pdf.section_title("5. Defect Frame Snapshots")

        per_row = 3
        img_w   = 56
        img_h   = 42
        padding = 4
        x_start = 12

        for i, e in enumerate(snap_events[:30]):
            col = i % per_row
            if col == 0 and i > 0:
                pdf.ln(img_h + 16)
            x = x_start + col * (img_w + padding)
            y = pdf.get_y()

            if y + img_h + 16 > pdf.h - 20:
                pdf.add_page()
                y = pdf.get_y()

            pdf.image(e["frame_path"], x=x, y=y, w=img_w, h=img_h)

            caption_lines = [
                f"#{i+1} | {e['predicted_class']}",
                e["timestamp"],
                f"Conf: {e['confidence']*100:.1f}%  Bbox: {e['bbox']}",
            ]
            pdf.set_xy(x, y + img_h + 1)
            pdf.set_font("Helvetica", "", 6)
            for line in caption_lines:
                safe = line.encode("latin-1", errors="replace").decode("latin-1")
                pdf.cell(img_w, 3.5, safe, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            if col < per_row - 1:
                pdf.set_xy(x + img_w + padding, y)

        pdf.ln(img_h + 16)

    # ── save PDF ──────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"report_{session_id}.pdf")
    pdf.output(pdf_path)
    print(f"[Report] PDF saved -> {pdf_path}")

    # ── CSV ───────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, f"report_{session_id}.csv")
    export_csv(reported_events, csv_path)
    print(f"[Report] CSV saved -> {csv_path}")

    # cleanup temp chart images
    for p in [bar_chart_path, timeline_chart_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    return pdf_path


# CLI:  python report_generator.py sessions/session_XYZ/session_log.json
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python report_generator.py <path/to/session_log.json>")
        sys.exit(1)
    log_path = sys.argv[1]
    out_dir  = os.path.dirname(log_path)
    generate_pdf_report(log_path, out_dir)
