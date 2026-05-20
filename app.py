"""
app.py  –  DriveSense AI  |  Main Application
==============================================
Run:
    python app.py

Gradio UI with:
  • Video upload + settings panel
  • Custom braking profile (vehicle type + 0→100 braking time)
  • Speed limit selector
  • Progress bar during processing
  • Annotated video output
  • Markdown safety report
  • Score breakdown with colour-coded gauges
"""

import cv2
import gradio as gr
import numpy as np
import os
import subprocess
import tempfile
from typing import Optional
from detector import DriveSenseDetector, DriverProfile, VEHICLE_FRONT_LENGTH_M, VEHICLE_DATABASE
from scorer  import SafetyScorer
from user_profile import UserProfile

# ─────────────────────────────────────────────
#  PROCESSING PIPELINE
# ─────────────────────────────────────────────

DESIRED_FPS   = 12     # output fps (frame-skipping for speed)
MODEL_NAME    = "yolo11n.pt"

# Green-light stationary threshold in SECONDS → converted to frames inside loop
GREEN_STATIONARY_S = 2.0

# Initialize user profile (handles multi-vehicle storage)
user_profile = UserProfile()


def _reencode_for_browser(raw_path: str) -> str:
    """
    Re-encode the mp4v output to H.264 using ffmpeg so it plays in all browsers.
    Falls back to the raw path if ffmpeg is not available.
    """
    out_path = raw_path.replace(".mp4", "_h264.mp4")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", raw_path,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "23",
                "-movflags", "+faststart",
                out_path,
            ],
            capture_output=True, timeout=300,
        )
        if result.returncode == 0 and os.path.exists(out_path):
            os.remove(raw_path)
            return out_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return raw_path   # ffmpeg not available — return raw (still works locally)


def process_video(
    input_video    : str,
    vehicle_name   : str,
    braking_time_s : float,
    speed_limit    : float,
    progress       = gr.Progress(track_tqdm=False),
) -> tuple[Optional[str], str, str]:
    """
    Main pipeline function called by Gradio.

    Returns:
        (output_video_path, markdown_report, score_summary_html)
    """
    if input_video is None:
        return None, "⚠️ Please upload a video file first.", ""

    # ── Build driver profile ──
    profile = DriverProfile(
        vehicle_name    = vehicle_name,
        braking_100_sec = braking_time_s,
        speed_limit_kmh = speed_limit,
    )

    detector = DriveSenseDetector(profile=profile, model_name=MODEL_NAME)
    scorer   = SafetyScorer(profile=profile)

    # ── Open video ──
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        return None, "❌ Could not open video file. Please check the format.", ""

    orig_fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_fps   = max(1.0, orig_fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Skip interval to match desired fps
    interval   = max(1, round(orig_fps / DESIRED_FPS))

    # Green-light threshold in frames (at the *output* fps)
    green_thresh = max(1, round(GREEN_STATIONARY_S * DESIRED_FPS))

    # ── Output video writer ──
    out_path = os.path.join(tempfile.gettempdir(), "drivesense_output.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    out      = cv2.VideoWriter(out_path, fourcc, DESIRED_FPS, (width, height))

    events   = []
    frame_idx = 0
    written   = 0
    progress(0, desc="Initialising model…")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            event = detector.analyse_frame(
                frame, frame_idx,
                fps=orig_fps,
                green_frames_threshold=green_thresh,
            )
            events.append(event)

            if event.annotated_frame is not None:
                out.write(event.annotated_frame)
            else:
                out.write(frame)

            written += 1
            pct = min(0.99, frame_idx / max(total_frames, 1))
            progress(pct, desc=f"Analysing frame {frame_idx}/{total_frames}…")

        frame_idx += 1

    cap.release()
    out.release()
    progress(1.0, desc="Generating report…")

    # Re-encode to H.264 for browser playback
    out_path = _reencode_for_browser(out_path)

    # ── Score ──
    report = scorer.score(events)

    # ── Record usage in user profile ──
    user_profile.record_usage(vehicle_name)

    # ── HTML score card ──
    html = _build_score_html(report)

    return out_path, report.markdown, html


# ─────────────────────────────────────────────
#  SCORE HTML
# ─────────────────────────────────────────────
def _score_color(s: float) -> str:
    if s >= 8:  return "#22c55e"
    if s >= 6:  return "#f59e0b"
    return "#ef4444"


def _build_score_html(report) -> str:
    def gauge(label, score):
        pct   = score * 10
        color = _score_color(score)
        return f"""
<div style="margin:10px 0;">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span style="font-family:'Courier New',monospace;font-size:13px;color:#ccc;">{label}</span>
    <span style="font-family:'Courier New',monospace;font-size:13px;color:{color};font-weight:bold;">{score:.1f}/10</span>
  </div>
  <div style="background:#1e1e2e;border-radius:6px;height:10px;overflow:hidden;">
    <div style="width:{pct}%;height:100%;background:{color};border-radius:6px;
                transition:width 0.8s ease;"></div>
  </div>
</div>"""

    overall_color = _score_color(report.overall_score)
    grade_map = [(9,"A+"),(8,"A"),(7,"B"),(6,"C"),(5,"D"),(0,"F")]
    grade = next(g for t,g in grade_map if report.overall_score >= t)

    html = f"""
<div style="background:#0f0f1a;border:1px solid #2a2a3e;border-radius:12px;
            padding:22px 28px;font-family:'Courier New',monospace;color:#e0e0e0;max-width:520px;">

  <div style="text-align:center;margin-bottom:20px;">
    <div style="font-size:11px;letter-spacing:3px;color:#666;margin-bottom:6px;">SAFETY SCORE</div>
    <div style="font-size:56px;font-weight:900;color:{overall_color};line-height:1;">{report.overall_score}</div>
    <div style="font-size:18px;color:{overall_color};margin-top:4px;">/ 10 &nbsp;({grade})</div>
    <div style="font-size:11px;color:#555;margin-top:8px;">
      {report.duration_s:.1f}s  ·  {report.analysed_frames} frames  ·  {report.vehicle_name}
    </div>
  </div>

  <div style="border-top:1px solid #2a2a3e;padding-top:16px;">
    {gauge("🚘  Following Distance", report.distance_score)}
    {gauge("🛣️  Lane Discipline",    report.lane_score)}
    {gauge("🚦  Traffic Lights",     report.light_score)}
  </div>

  <div style="border-top:1px solid #2a2a3e;padding-top:14px;margin-top:6px;
              font-size:12px;color:#888;display:grid;grid-template-columns:1fr 1fr;gap:6px;">
    <div>Too-close frames: <b style="color:#ccc;">{report.too_close_count}</b></div>
    <div>Lane drifts: <b style="color:#ccc;">{report.drift_events}</b></div>
    <div>Over-line events: <b style="color:#ccc;">{report.over_line_events}</b></div>
    <div>Light violations: <b style="color:#ccc;">{report.light_violations}</b></div>
  </div>

</div>"""
    return html


# ─────────────────────────────────────────────
#  GRADIO UI  –  VEHICLE PRESETS & SETUP
# ─────────────────────────────────────────────

# Build presets from VEHICLE_DATABASE + user-stored vehicles + custom option
# This ensures UI stays in sync with detector.py database
def build_vehicle_presets():
    """
    Dynamically build vehicle preset list from database.
    Includes user's stored vehicles with usage counts for easy access.
    """
    presets = {}
    
    # User's saved vehicles (show first, with usage info)
    user_vehicles = user_profile.get_vehicle_list()
    if user_vehicles:
        for vname in user_vehicles:
            usage = user_profile.vehicles[vname].get("usage_count", 0)
            braking = user_profile.vehicles[vname].get("braking_time", 3.5)
            label = f"📌 {vname} (used {usage} times)"
            presets[label] = (vname, braking)
        
        # Separator
        presets["─" * 40] = ("Standard Car", 3.5)
    
    # All database vehicles (sorted)
    for vname in sorted(VEHICLE_DATABASE.keys()):
        if vname not in user_profile.vehicles:  # Don't duplicate
            presets[vname] = (vname, 3.5)
    
    # Custom option at end
    presets["🔧 Custom (enter below)"] = ("Custom Vehicle", 3.5)
    
    return presets

VEHICLE_PRESETS = build_vehicle_presets()

CSS = """
/* ── Global ── */
body, .gradio-container { background:#0a0a14 !important; }
.gradio-container { max-width:1100px !important; margin:0 auto; }

/* ── Panel cards ── */
.panel-card {
    background:#10101e;
    border:1px solid #1e1e35;
    border-radius:14px;
    padding:18px 22px;
}

/* ── Header ── */
.ds-header {
    text-align:center;
    padding:32px 0 24px;
    letter-spacing:0.05em;
}
.ds-header h1 {
    font-family:'Courier New',monospace;
    font-size:2.2rem;
    font-weight:900;
    color:#e8e8ff;
    margin:0;
    letter-spacing:0.12em;
}
.ds-header p {
    font-family:'Courier New',monospace;
    color:#555;
    font-size:0.8rem;
    margin:6px 0 0;
    letter-spacing:0.2em;
    text-transform:uppercase;
}

/* ── Buttons ── */
button.primary-btn, .gr-button-primary {
    background: linear-gradient(135deg,#3b5bdb,#7048e8) !important;
    border:none !important;
    font-family:'Courier New',monospace !important;
    letter-spacing:0.1em !important;
    font-weight:700 !important;
    border-radius:8px !important;
    color:#fff !important;
}
button.primary-btn:hover {
    opacity:0.88 !important;
}

/* ── Labels ── */
label span { font-family:'Courier New',monospace !important; color:#999 !important; font-size:12px !important; letter-spacing:0.08em !important; }
"""

def update_braking_from_preset(preset_label):
    """
    When a preset is chosen, auto-fill the braking time.
    If user has stored this vehicle, use their recorded braking time.
    Otherwise use the database default (3.5s).
    """
    vname, braking = VEHICLE_PRESETS.get(preset_label, ("Custom Vehicle", 3.5))
    
    # If this vehicle is in user's stored collection, use their braking time
    if vname in user_profile.vehicles:
        braking = user_profile.vehicles[vname].get("braking_time", braking)
    
    return braking, vname


def get_suggested_vehicle() -> str:
    """
    Return suggested vehicle for next upload.
    If user has only one vehicle or a clear primary, suggest it.
    Otherwise return first database vehicle.
    """
    suggested = user_profile.get_suggested_vehicle()
    if suggested:
        for label, (vname, _) in VEHICLE_PRESETS.items():
            if vname == suggested:
                return label
    # Default to first preset
    return next(iter(VEHICLE_PRESETS.keys()))


with gr.Blocks(css=CSS, title="DriveSense AI") as demo:

    # ── Header ──────────────────────────────────
    gr.HTML("""
    <div class="ds-header">
      <h1>⬡ DRIVESENSE AI</h1>
      <p>Intelligent Dashcam Analysis · Safety Scoring · Driver Feedback</p>
    </div>
    """)

    with gr.Row():

        # ── Left column: inputs ──────────────────
        with gr.Column(scale=1):
            gr.HTML('<div class="panel-card">')
            gr.Markdown("### 📹 Upload Footage")
            video_input = gr.Video(
                label="Dashcam / Wearable Camera Video",
                height=220,
            )
            gr.HTML('</div><br>')

            gr.HTML('<div class="panel-card">')
            gr.Markdown("### ⚙️ Driver & Vehicle Settings")

            vehicle_preset = gr.Dropdown(
                choices=list(VEHICLE_PRESETS.keys()),
                value=get_suggested_vehicle(),
                label="🚗 Vehicle Preset  (saved vehicles shown first)",
            )
            vehicle_name_box = gr.Textbox(
                value=VEHICLE_PRESETS[get_suggested_vehicle()][0],
                label="Vehicle Name (shown in report) — auto-filled from database",
                max_lines=1,
            )
            braking_slider = gr.Slider(
                minimum=2.0, maximum=8.0, value=VEHICLE_PRESETS[get_suggested_vehicle()][1], step=0.1,
                label="0–100 km/h Braking Time (seconds) — affects safe distance",
            )
            speed_limit_slider = gr.Slider(
                minimum=30, maximum=130, value=70, step=5,
                label="Road Speed Limit (km/h)",
            )
            gr.HTML('</div><br>')

            gr.HTML("""
            <div class="panel-card" style="font-family:'Courier New',monospace;
                 font-size:10px;color:#666;line-height:1.9;">
              <b style="color:#aaa;">🔬 TRIANGLE SIMILARITY DISTANCE CALCULATION</b><br>
              <span style="color:#777;">
                <b>Method:</b> distance = (focal_const × vehicle_width) / bbox_width_px<br>
                <b>Advantage:</b> Robust on hills & curves (uses width, not Y-coordinate)<br>
                <b>Database:</b> Vehicle-specific dimensions (PakWheels specs)<br>
              </span><br>
              <b style="color:#aaa;">SAFE DISTANCE FORMULA</b><br>
              d = v×t<sub>reaction</sub> + v²/(2a) × 1.2<br>
              where t<sub>reaction</sub>=1.5s,  a from your braking profile.<br><br>
              <b style="color:#aaa;">SAFETY SCORE WEIGHTS</b><br>
              Following Distance · · · · 40%<br>
              Lane Discipline · · · · · 35%<br>
              Traffic Light Response · 25%
            </div>
            """)

            analyse_btn = gr.Button(
                "▶  ANALYSE FOOTAGE",
                variant="primary",
                elem_classes=["primary-btn"],
            )

        # ── Right column: outputs ────────────────
        with gr.Column(scale=2):
            gr.HTML('<div class="panel-card">')
            gr.Markdown("### 🎬 Annotated Output")
            video_output = gr.Video(label="Processed Video", height=320)
            gr.HTML('</div><br>')

            gr.HTML('<div class="panel-card">')
            gr.Markdown("### 📊 Score Card")
            score_html = gr.HTML()
            gr.HTML('</div><br>')

            gr.HTML('<div class="panel-card">')
            gr.Markdown("### 📋 Full Safety Report")
            report_md = gr.Markdown()
            gr.HTML('</div>')

    # ── Wire preset → braking slider ──────────────
    vehicle_preset.change(
        fn=update_braking_from_preset,
        inputs=[vehicle_preset],
        outputs=[braking_slider, vehicle_name_box],
    )

    # ── Main analyse button ────────────────────────
    analyse_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            vehicle_name_box,
            braking_slider,
            speed_limit_slider,
        ],
        outputs=[video_output, report_md, score_html],
    )

    # ── Footer ────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;padding:28px 0 12px;
         font-family:'Courier New',monospace;font-size:11px;
         color:#333;letter-spacing:0.1em;">
      DRIVESENSE AI · FOR EDUCATIONAL USE · ITU LAHORE BSAI
    </div>
    """)


#  ENTRY POINT
if __name__ == "__main__":
    demo.launch(
        share=True,           # generates a public Gradio URL
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )