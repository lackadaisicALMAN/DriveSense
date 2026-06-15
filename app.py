"""
app.py  –  DriveSense AI  |  Main Web Dashboard
=============================================
Run:
    python app.py

A premium, high-tech, responsive Gradio Dashboard featuring:
  • Single Drive Safety Analysis with real-time gauges and advice
  • 1v1 Driving Battle Mode (compete against a friend or past runs)
  • Matplotlib Following Distance Telemetry Charting
  • Session History Logs & Comparators
  • Saved Vehicle Configurations
"""

import cv2
import gradio as gr
import numpy as np
import os
import subprocess
import tempfile
import time
from typing import Optional

from detector import DriveSenseDetector, DriverProfile, VEHICLE_FRONT_LENGTH_M, VEHICLE_DATABASE
from scorer import SafetyScorer, SafetyReport
from user_profile import UserProfile
from analytics import generate_distance_chart

# ─────────────────────────────────────────────
#  PIPELINE CONFIGURATION
# ─────────────────────────────────────────────
DESIRED_FPS        = 12
MODEL_NAME         = "yolo11n.pt"
GREEN_STATIONARY_S = 2.0

# Initialise persistent user profile
user_profile = UserProfile()


def _reencode_for_browser(raw_path: str) -> str:
    """
    Re-encode the OpenCV output video to H.264 using ffmpeg.
    Required for playback in HTML5 browser elements.
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
    return raw_path


def _run_pipeline(
    input_video    : str,
    vehicle_name   : str,
    braking_time_s : float,
    speed_limit    : float,
    processing_speed: str,
    progress,
    offset_pct     : float = 0.0,
    total_pct      : float = 1.0,
) -> tuple[str, SafetyReport, list]:
    """Runs the core CV processing pipeline over a video."""
    profile = DriverProfile(
        vehicle_name    = vehicle_name,
        braking_100_sec = braking_time_s,
        speed_limit_kmh = speed_limit,
    )

    detector = DriveSenseDetector(profile=profile, model_name=MODEL_NAME)
    scorer   = SafetyScorer(profile=profile)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    # Map processing speed to target FPS
    fps_map = {
        "⚡ Fast (5 FPS)": 5.0,
        "⚖️ Standard (10 FPS)": 10.0,
        "🎯 High Precision (15 FPS)": 15.0
    }
    target_fps = fps_map.get(processing_speed, 10.0)

    orig_fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_fps   = max(1.0, orig_fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale resolution down to max 1280px width to speed up CPU inference & resizing
    max_w = 1280
    scale_factor = 1.0
    if width > max_w:
        scale_factor = max_w / width
        width = max_w
        height = int(height * scale_factor)

    interval   = max(1, round(orig_fps / target_fps))
    output_fps = orig_fps / interval
    green_thresh = max(1, round(GREEN_STATIONARY_S * output_fps))

    out_dir = tempfile.gettempdir()
    out_name = f"drivesense_out_{int(time.time())}_{os.path.basename(input_video)}"
    out_path = os.path.join(out_dir, out_name)
    
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    out      = cv2.VideoWriter(out_path, fourcc, output_fps, (width, height))

    events   = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            if scale_factor < 1.0:
                frame = cv2.resize(frame, (width, height))
                
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

            pct = offset_pct + (frame_idx / max(total_frames, 1)) * total_pct
            progress(min(0.99, pct), desc=f"Processing frames ({int(pct*100)}%)...")

        frame_idx += 1

    cap.release()
    out.release()
    
    out_path = _reencode_for_browser(out_path)
    report = scorer.score(events)
    
    return out_path, report, events


def process_video(
    input_video    : str,
    vehicle_name   : str,
    braking_time_s : float,
    speed_limit    : float,
    processing_speed: str,
    progress       = gr.Progress(track_tqdm=False),
):
    """Gradio entrypoint for analyzing a single drive."""
    if input_video is None:
        return None, "⚠️ Please upload a video file first.", "", None, gr.update(), ""

    try:
        out_path, report, events = _run_pipeline(
            input_video, vehicle_name, braking_time_s, speed_limit, processing_speed, progress, 0.0, 1.0
        )
    except Exception as e:
        return None, f"❌ Error processing video: {str(e)}", "", None, gr.update(), ""

    # Save details to profile database
    user_profile.record_usage(vehicle_name)
    user_profile.record_run(
        video_name   = input_video,
        vehicle_name = vehicle_name,
        score        = report.overall_score,
        dist_score   = report.distance_score,
        lane_score   = report.lane_score,
        light_score  = report.light_score,
        duration_s   = report.duration_s,
        too_close    = report.too_close_count,
        drifts       = report.drift_events,
        over_lines   = report.over_line_events,
        light_viols  = report.light_violations,
        stop_viols   = report.stop_sign_violations,
        collisions   = report.collision_warnings,
    )

    # Generate telemetry chart
    v = speed_limit / 3.6
    t_reaction = 1.5
    v100 = 100 / 3.6
    a = v100 / max(braking_time_s, 0.5)
    safe_d = (v * t_reaction + (v ** 2) / (2 * a)) * 1.2
    chart_path = generate_distance_chart(events, safe_d)

    score_html = _build_score_html(report)
    history_choices, history_table = load_history_ui()

    return (
        out_path, 
        report.markdown, 
        score_html, 
        chart_path, 
        gr.update(choices=history_choices), 
        history_table
    )


def process_1v1_battle(
    video_1: str, name_1: str, vehicle_1: str, braking_1: float,
    video_2: str, name_2: str, vehicle_2: str, braking_2: float,
    speed_limit: float,
    processing_speed: str,
    progress = gr.Progress(track_tqdm=False)
):
    """Gradio entrypoint for 1v1 battle mode."""
    if not video_1 or not video_2:
        return None, None, "⚠️ Please upload both videos to start the battle.", "", gr.update(), ""
        
    name_1 = name_1 or "Driver 1"
    name_2 = name_2 or "Driver 2"

    try:
        # Driver 1 (0% to 50%)
        out_1, report_1, _ = _run_pipeline(
            video_1, vehicle_1, braking_1, speed_limit, processing_speed, progress, 0.0, 0.5
        )
        user_profile.record_usage(vehicle_1)
        user_profile.record_run(
            video_name   = video_1,
            vehicle_name = f"{name_1} ({vehicle_1})",
            score        = report_1.overall_score,
            dist_score   = report_1.distance_score,
            lane_score   = report_1.lane_score,
            light_score  = report_1.light_score,
            duration_s   = report_1.duration_s,
            too_close    = report_1.too_close_count,
            drifts       = report_1.drift_events,
            over_lines   = report_1.over_line_events,
            light_viols  = report_1.light_violations,
            stop_viols   = report_1.stop_sign_violations,
            collisions   = report_1.collision_warnings,
        )

        # Driver 2 (50% to 100%)
        out_2, report_2, _ = _run_pipeline(
            video_2, vehicle_2, braking_2, speed_limit, processing_speed, progress, 0.5, 0.5
        )
        user_profile.record_usage(vehicle_2)
        user_profile.record_run(
            video_name   = video_2,
            vehicle_name = f"{name_2} ({vehicle_2})",
            score        = report_2.overall_score,
            dist_score   = report_2.distance_score,
            lane_score   = report_2.lane_score,
            light_score  = report_2.light_score,
            duration_s   = report_2.duration_s,
            too_close    = report_2.too_close_count,
            drifts       = report_2.drift_events,
            over_lines   = report_2.over_line_events,
            light_viols  = report_2.light_violations,
            stop_viols   = report_2.stop_sign_violations,
            collisions   = report_2.collision_warnings,
        )
        
    except Exception as e:
        return None, None, f"❌ Error processing battle: {str(e)}", "", gr.update(), ""

    battle_html = _build_battle_html(name_1, report_1, name_2, report_2)
    battle_md = _build_battle_markdown(name_1, report_1, name_2, report_2)
    
    choices, history_table = load_history_ui()
    
    return out_1, out_2, battle_html, battle_md, gr.update(choices=choices), history_table


# ─────────────────────────────────────────────
#  HTML & MARKDOWN GENERATION
# ─────────────────────────────────────────────
def _score_color(s: float) -> str:
    if s >= 8:  return "#22c55e" # Neon Green
    if s >= 6:  return "#eab308" # Neon Yellow
    return "#ef4444"             # Neon Red


def _build_score_html(report: SafetyReport) -> str:
    def gauge(label, score):
        pct   = score * 10
        color = _score_color(score)
        return f"""
<div style="margin:12px 0;">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span style="font-family:'Courier New',monospace;font-size:12px;color:#aaa;">{label}</span>
    <span style="font-family:'Courier New',monospace;font-size:12px;color:{color};font-weight:bold;">{score:.1f}/10</span>
  </div>
  <div style="background:#1e1e2f;border-radius:6px;height:10px;overflow:hidden;border:1px solid #26263e;">
    <div style="width:{pct}%;height:100%;background:{color};border-radius:6px;
                box-shadow: 0 0 8px {color}88; transition:width 0.8s ease;"></div>
  </div>
</div>"""

    overall_color = _score_color(report.overall_score)
    grade_map = [(9,"A+"),(8,"A"),(7,"B"),(6,"C"),(5,"D"),(0,"F")]
    grade = next(g for t,g in grade_map if report.overall_score >= t)

    html = f"""
<div style="background:#0f0f1c;border:1px solid #2a2a44;border-radius:12px;
            padding:24px 28px;font-family:'Courier New',monospace;color:#e0e0f5;max-width:100%;">

  <div style="text-align:center;margin-bottom:20px;">
    <div style="font-size:11px;letter-spacing:4px;color:#888;margin-bottom:6px;">SAFETY SCORE</div>
    <div style="font-size:60px;font-weight:900;color:{overall_color};line-height:1;
                text-shadow: 0 0 15px {overall_color}55;">{report.overall_score}</div>
    <div style="font-size:18px;color:{overall_color};margin-top:4px;">/ 10 &nbsp;({grade})</div>
    <div style="font-size:11px;color:#777;margin-top:8px;">
      {report.duration_s:.1f}s  ·  {report.analysed_frames} frames  ·  {report.vehicle_name}
    </div>
  </div>

  <div style="border-top:1px solid #26263e;padding-top:16px;">
    {gauge("🚘  Following Gap & TTC", report.distance_score)}
    {gauge("🛣️  Lane Keeping Stability", report.lane_score)}
    {gauge("🚦  Intersection Response", report.light_score)}
  </div>

  <div style="border-top:1px solid #26263e;padding-top:14px;margin-top:10px;
              font-size:11px;color:#888;display:grid;grid-template-columns:1fr 1fr;gap:8px 16px;">
    <div>Too-close frames: <b style="color:#ccc;">{report.too_close_count}</b></div>
    <div>Collision alerts: <b style="color:#ccc;">{report.collision_warnings}</b></div>
    <div>Lane drifts: <b style="color:#ccc;">{report.drift_events}</b></div>
    <div>Over-line events: <b style="color:#ccc;">{report.over_line_events}</b></div>
    <div>Signal delays: <b style="color:#ccc;">{report.light_violations}</b></div>
    <div>Stop sign rolls: <b style="color:#ccc;">{report.stop_sign_violations}</b></div>
  </div>

</div>"""
    return html


def _build_battle_html(name_1: str, report_1, name_2: str, report_2) -> str:
    s1 = report_1.overall_score
    s2 = report_2.overall_score
    
    if s1 > s2:
        winner_text = f"🏆 {name_1} wins! ({s1:.1f} vs {s2:.1f})"
        w_color = "#22c55e"
    elif s2 > s1:
        winner_text = f"🏆 {name_2} wins! ({s2:.1f} vs {s1:.1f})"
        w_color = "#22c55e"
    else:
        winner_text = f"👔 It's a Tie! ({s1:.1f} vs {s2:.1f})"
        w_color = "#eab308"
        
    def score_badge(score):
        color = _score_color(score)
        return f'<span style="color:{color};font-weight:bold;font-size:24px;text-shadow:0 0 8px {color}44;">{score:.1f}/10</span>'
        
    html = f"""
<div style="background:#0f0f1c;border:1px solid #2a2a44;border-radius:12px;
            padding:24px;font-family:'Courier New',monospace;color:#e0e0f5;max-width:100%;">

  <div style="text-align:center;margin-bottom:24px;padding:14px;background:#151528;border-radius:8px;border:1px solid #303054;">
    <div style="font-size:11px;letter-spacing:4px;color:#888;margin-bottom:6px;">BATTLE OUTCOME</div>
    <div style="font-size:32px;font-weight:900;color:{w_color};text-shadow:0 0 15px {w_color}55;">{winner_text}</div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;">
    <!-- Driver 1 -->
    <div style="background:#151528;border:1px solid #26263e;border-radius:8px;padding:16px;text-align:center;">
      <div style="font-size:20px;font-weight:bold;color:#4c6ef5;margin-bottom:4px;">{name_1}</div>
      <div style="font-size:11px;color:#888;margin-bottom:12px;">{report_1.vehicle_name}</div>
      <div>{score_badge(s1)}</div>
      <div style="margin-top:14px;text-align:left;font-size:11px;line-height:1.7;">
        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #222;padding:2px 0;"><span>Gap Score:</span><b style="color:#ccc;">{report_1.distance_score:.1f}</b></div>
        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #222;padding:2px 0;"><span>Lane Score:</span><b style="color:#ccc;">{report_1.lane_score:.1f}</b></div>
        <div style="display:flex;justify-content:space-between;padding:2px 0;"><span>Intersection:</span><b style="color:#ccc;">{report_1.light_score:.1f}</b></div>
      </div>
    </div>
    
    <!-- Driver 2 -->
    <div style="background:#151528;border:1px solid #26263e;border-radius:8px;padding:16px;text-align:center;">
      <div style="font-size:20px;font-weight:bold;color:#7048e8;margin-bottom:4px;">{name_2}</div>
      <div style="font-size:11px;color:#888;margin-bottom:12px;">{report_2.vehicle_name}</div>
      <div>{score_badge(s2)}</div>
      <div style="margin-top:14px;text-align:left;font-size:11px;line-height:1.7;">
        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #222;padding:2px 0;"><span>Gap Score:</span><b style="color:#ccc;">{report_2.distance_score:.1f}</b></div>
        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #222;padding:2px 0;"><span>Lane Score:</span><b style="color:#ccc;">{report_2.lane_score:.1f}</b></div>
        <div style="display:flex;justify-content:space-between;padding:2px 0;"><span>Intersection:</span><b style="color:#ccc;">{report_2.light_score:.1f}</b></div>
      </div>
    </div>
  </div>

  <div style="border-top:1px solid #26263e;padding-top:16px;">
    <div style="font-size:14px;font-weight:bold;color:#aaa;margin-bottom:12px;text-align:center;letter-spacing:2px;">DRIVING ENVELOPE METRICS</div>
    
    <table style="width:100%;font-size:12px;border-collapse:collapse;text-align:left;">
      <thead>
        <tr style="border-bottom:1px solid #26263e;color:#888;height:30px;">
          <th style="padding:6px 0;">Driving Metric</th>
          <th style="padding:6px 0;text-align:center;color:#4c6ef5;width:30%;">{name_1}</th>
          <th style="padding:6px 0;text-align:center;color:#7048e8;width:30%;">{name_2}</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom:1px solid #1a1a2e;height:32px;">
          <td style="padding:6px 0;color:#aaa;">Tailgate Frames</td>
          <td style="padding:6px 0;text-align:center;color:{'#ef4444' if report_1.too_close_count > report_2.too_close_count else '#aaa'}">{report_1.too_close_count}</td>
          <td style="padding:6px 0;text-align:center;color:{'#ef4444' if report_2.too_close_count > report_1.too_close_count else '#aaa'}">{report_2.too_close_count}</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e;height:32px;">
          <td style="padding:6px 0;color:#aaa;">Collision Alerts</td>
          <td style="padding:6px 0;text-align:center;color:{'#ef4444' if report_1.collision_warnings > report_2.collision_warnings else '#aaa'}">{report_1.collision_warnings}</td>
          <td style="padding:6px 0;text-align:center;color:{'#ef4444' if report_2.collision_warnings > report_1.collision_warnings else '#aaa'}">{report_2.collision_warnings}</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e;height:32px;">
          <td style="padding:6px 0;color:#aaa;">Lane Drifts</td>
          <td style="padding:6px 0;text-align:center;color:{'#ef4444' if report_1.drift_events > report_2.drift_events else '#aaa'}">{report_1.drift_events}</td>
          <td style="padding:6px 0;text-align:center;color:{'#ef4444' if report_2.drift_events > report_1.drift_events else '#aaa'}">{report_2.drift_events}</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e;height:32px;">
          <td style="padding:6px 0;color:#aaa;">Lane Straddles</td>
          <td style="padding:6px 0;text-align:center;color:{'#ef4444' if report_1.over_line_events > report_2.over_line_events else '#aaa'}">{report_1.over_line_events}</td>
          <td style="padding:6px 0;text-align:center;color:{'#ef4444' if report_2.over_line_events > report_1.over_line_events else '#aaa'}">{report_2.over_line_events}</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e;height:32px;">
          <td style="padding:6px 0;color:#aaa;">Signal Violations</td>
          <td style="padding:6px 0;text-align:center;color:{'#ef4444' if report_1.light_violations > report_2.light_violations else '#aaa'}">{report_1.light_violations}</td>
          <td style="padding:6px 0;text-align:center;color:{'#ef4444' if report_2.light_violations > report_1.light_violations else '#aaa'}">{report_2.light_violations}</td>
        </tr>
        <tr style="border-bottom:1px solid #26263e;height:32px;">
          <td style="padding:6px 0;color:#aaa;">Stop Sign Rolls</td>
          <td style="padding:6px 0;text-align:center;color:{'#ef4444' if report_1.stop_sign_violations > report_2.stop_sign_violations else '#aaa'}">{report_1.stop_sign_violations}</td>
          <td style="padding:6px 0;text-align:center;color:{'#ef4444' if report_2.stop_sign_violations > report_1.stop_sign_violations else '#aaa'}">{report_2.stop_sign_violations}</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>
"""
    return html


def _build_battle_markdown(name_1: str, report_1, name_2: str, report_2) -> str:
    s1 = report_1.overall_score
    s2 = report_2.overall_score
    
    if s1 > s2:
        winner = name_1
        loser = name_2
        diff = s1 - s2
        summary = f"🏆 **{winner}** outperformed **{loser}** by **{diff:.1f}** safety points, demonstrating superior road discipline."
    elif s2 > s1:
        winner = name_2
        loser = name_1
        diff = s2 - s1
        summary = f"🏆 **{winner}** outperformed **{loser}** by **{diff:.1f}** safety points, demonstrating superior road discipline."
    else:
        summary = "👔 Both drivers showed identical safety ratings in this match. Excellent job maintaining a steady safety envelope!"

    lines = [
        f"# ⚔️ 1v1 Battle Scoreboard: {name_1} vs {name_2}",
        f"",
        summary,
        f"",
        f"### Summary of infractions:",
        f"- **{name_1}** driving **{report_1.vehicle_name}** logged:",
        f"  - Tailgating events: **{report_1.too_close_count}** frames",
        f"  - Forward Collision warnings: **{report_1.collision_warnings}** events",
        f"  - Lane discipline issues: **{report_1.drift_events + report_1.over_line_events}** times",
        f"  - Intersection infractions: **{report_1.light_violations + report_1.stop_sign_violations}** times",
        f"",
        f"- **{name_2}** driving **{report_2.vehicle_name}** logged:",
        f"  - Tailgating events: **{report_2.too_close_count}** frames",
        f"  - Forward Collision warnings: **{report_2.collision_warnings}** events",
        f"  - Lane discipline issues: **{report_2.drift_events + report_2.over_line_events}** times",
        f"  - Intersection infractions: **{report_2.light_violations + report_2.stop_sign_violations}** times",
        f"",
        f"---",
        f"*Generated by DriveSense AI 1v1 Arena.*"
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  HISTORY AND PRESETS
# ─────────────────────────────────────────────
def load_history_ui() -> tuple[list, str]:
    """Load history and format it into a table and dropdown choices."""
    history = user_profile.get_history()
    if not history:
        return [], "<div style='text-align:center;padding:30px;color:#555;font-family:monospace;'>No driving records logged. Analyze a video to get started!</div>"
        
    html = """
<div style="font-family:'Courier New',monospace;color:#e0e0f5;max-width:100%;">
  <table style="width:100%;font-size:12px;border-collapse:collapse;text-align:left;">
    <thead>
      <tr style="border-bottom:1px solid #26263e;color:#888;height:32px;">
        <th style="padding:8px 0;">Timestamp</th>
        <th style="padding:8px 0;">Session Title (Video Source)</th>
        <th style="padding:8px 0;">Vehicle used</th>
        <th style="padding:8px 0;text-align:center;">Safety Rating</th>
      </tr>
    </thead>
    <tbody>
    """
    choices = []
    for idx, run in enumerate(history):
        score_color = _score_color(run["overall_score"])
        html += f"""
      <tr style="border-bottom:1px solid #141424;height:36px;transition: background 0.2s;">
        <td style="color:#777;padding:8px 0;">{run["timestamp"]}</td>
        <td style="color:#ccc;font-weight:bold;padding:8px 0;">{run["video_name"]}</td>
        <td style="color:#aaa;padding:8px 0;">{run["vehicle_name"]}</td>
        <td style="text-align:center;color:{score_color};font-weight:bold;font-size:13px;padding:8px 0;">{run["overall_score"]:.1f}/10</td>
      </tr>
        """
        label = f"{run['timestamp']} - {run['video_name']} ({run['overall_score']:.1f}/10)"
        choices.append((label, run["id"]))
        
    html += """
    </tbody>
  </table>
</div>
    """
    return choices, html


def build_vehicle_presets():
    """Dynamically build presets list from database & user saved vehicles."""
    presets = {}
    
    # Stored vehicles
    user_vehicles = user_profile.get_vehicle_list()
    if user_vehicles:
        for vname in user_vehicles:
            usage = user_profile.vehicles[vname].get("usage_count", 0)
            braking = user_profile.vehicles[vname].get("braking_time", 3.5)
            label = f"📌 {vname} (used {usage} times)"
            presets[label] = (vname, braking)
        presets["─" * 40] = ("Standard Car", 3.5)
    
    # Default database
    for vname in sorted(VEHICLE_DATABASE.keys()):
        if vname not in user_profile.vehicles:
            presets[vname] = (vname, 3.5)
    
    presets["🔧 Custom (enter below)"] = ("Custom Vehicle", 3.5)
    return presets

VEHICLE_PRESETS = build_vehicle_presets()


def update_braking_from_preset(preset_label):
    vname, braking = VEHICLE_PRESETS.get(preset_label, ("Custom Vehicle", 3.5))
    if vname in user_profile.vehicles:
        braking = user_profile.vehicles[vname].get("braking_time", braking)
    return braking, vname


def get_suggested_vehicle() -> str:
    suggested = user_profile.get_suggested_vehicle()
    if suggested:
        for label, (vname, _) in VEHICLE_PRESETS.items():
            if vname == suggested:
                return label
    return next(iter(VEHICLE_PRESETS.keys()))


# ─────────────────────────────────────────────
#  TAB ACTIONS AND HANDLERS
# ─────────────────────────────────────────────
def add_new_vehicle_action(name, braking):
    if not name or name.strip() == "":
        return gr.update(), "⚠️ Vehicle name cannot be blank."
    user_profile.add_vehicle(name, float(braking))
    
    # Rebuild preset choices in app
    global VEHICLE_PRESETS
    VEHICLE_PRESETS = build_vehicle_presets()
    new_choices = list(VEHICLE_PRESETS.keys())
    
    return gr.update(choices=new_choices, value=get_suggested_vehicle()), f"✅ Saved {name} to profile database."


def clear_history_action():
    user_profile.clear_history()
    choices, table = load_history_ui()
    return gr.update(choices=choices), table


# ─────────────────────────────────────────────
#  GRADIO CUSTOM STYLING (CSS)
# ─────────────────────────────────────────────
CSS = """
/* Dark cyberpunk glow theme */
body, .gradio-container {
    background-color: #05050d !important;
    background-image: radial-gradient(circle at 50% 50%, #120e26 0%, #05050d 100%) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif !important;
    color: #e2e2f0 !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
}

.panel-card {
    background: rgba(15, 15, 28, 0.72) !important;
    border: 1px solid #252542 !important;
    border-radius: 12px !important;
    padding: 22px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
    backdrop-filter: blur(10px);
}

.ds-header {
    text-align: center;
    padding: 30px 0 20px;
}
.ds-header h1 {
    font-family: 'Courier New', monospace;
    font-size: 2.5rem;
    font-weight: 900;
    color: #f3f3ff;
    text-shadow: 0 0 15px rgba(76, 110, 245, 0.5);
    margin: 0;
    letter-spacing: 0.15em;
}
.ds-header p {
    font-family: 'Courier New', monospace;
    color: #6a6a9d;
    font-size: 0.85rem;
    margin: 8px 0 0;
    letter-spacing: 0.25em;
    text-transform: uppercase;
}

button.primary-btn, .gr-button-primary {
    background: linear-gradient(135deg, #4c6ef5 0%, #7048e8 100%) !important;
    border: none !important;
    font-family: 'Courier New', monospace !important;
    letter-spacing: 0.15em !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(112, 72, 232, 0.4) !important;
    transition: all 0.3s ease !important;
}
button.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(112, 72, 232, 0.6) !important;
    opacity: 0.95 !important;
}

label span {
    font-family: 'Courier New', monospace !important;
    color: #8f8faf !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
}

.gr-tabs {
    border-bottom: 2px solid #22223a !important;
}
.tab-nav button {
    font-family: 'Courier New', monospace !important;
    font-size: 13px !important;
    letter-spacing: 0.08em !important;
    color: #6a6a9d !important;
    background: transparent !important;
    border: none !important;
}
.tab-nav button.selected {
    color: #e2e2f0 !important;
    border-bottom: 2px solid #4c6ef5 !important;
    font-weight: bold !important;
}
"""

# ─────────────────────────────────────────────
#  GRADIO APPLICATION LAYOUT
# ─────────────────────────────────────────────
with gr.Blocks(title="DriveSense AI Dashboard") as demo:

    # ── Header ──────────────────────────────────
    gr.HTML("""
    <div class="ds-header">
      <h1>⬡ DRIVESENSE AI</h1>
      <p>ADAS Driver Evaluation · Telemetry Charting · 1v1 Battle Arena</p>
    </div>
    """)

    # Setup starting UI choices
    initial_history_choices, initial_history_table = load_history_ui()

    with gr.Tabs():
        
        # ── TAB 1: SINGLE DRIVE ANALYZER ────────────
        with gr.Tab("🚗 Single Drive Evaluation"):
            with gr.Row():
                
                # Left inputs
                with gr.Column(scale=1):
                    gr.HTML('<div class="panel-card">')
                    gr.Markdown("### 📹 Upload Dashcam / Bumper Video")
                    video_input = gr.Video(
                        label="Source Footage (supports mp4/avi)",
                        height=210,
                    )
                    gr.HTML('</div><br>')

                    gr.HTML('<div class="panel-card">')
                    gr.Markdown("### ⚙️ Calibration Settings")

                    vehicle_preset = gr.Dropdown(
                        choices=list(VEHICLE_PRESETS.keys()),
                        value=get_suggested_vehicle(),
                        label="🚗 Vehicle Config Database",
                    )
                    vehicle_name_box = gr.Textbox(
                        value=VEHICLE_PRESETS[get_suggested_vehicle()][0],
                        label="Report Vehicle Identifier (Auto-filled)",
                        max_lines=1,
                    )
                    braking_slider = gr.Slider(
                        minimum=2.0, maximum=8.0, value=VEHICLE_PRESETS[get_suggested_vehicle()][1], step=0.1,
                        label="0–100 km/h Braking Profile (seconds)",
                    )
                    speed_limit_slider = gr.Slider(
                        minimum=30, maximum=130, value=70, step=5,
                        label="Road Speed Limit (km/h)",
                    )
                    processing_speed_dropdown = gr.Dropdown(
                        choices=["⚡ Fast (5 FPS)", "⚖️ Standard (10 FPS)", "🎯 High Precision (15 FPS)"],
                        value="⚖️ Standard (10 FPS)",
                        label="Processing Speed & Quality",
                    )
                    gr.HTML('</div><br>')

                    analyse_btn = gr.Button(
                        "▶  START EVALUATION",
                        variant="primary",
                        elem_classes=["primary-btn"],
                    )

                # Right outputs
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div class="panel-card">')
                            gr.Markdown("### 🎬 Annotated ADAS Stream")
                            video_output = gr.Video(label="ADAS Stream Overlay", height=320)
                            gr.HTML('</div>')
                            
                        with gr.Column(scale=1):
                            gr.HTML('<div class="panel-card">')
                            gr.Markdown("### 📊 Active Safety Scorecard")
                            score_html = gr.HTML(value="<div style='text-align:center;padding:60px;color:#444;font-family:monospace;'>Upload a video and start evaluation to display safety scorecard.</div>")
                            gr.HTML('</div>')
                    
                    gr.HTML('<br><div class="panel-card">')
                    gr.Markdown("### 📈 Following Gap Telemetry Profile")
                    telemetry_chart = gr.Image(label="Telemetry Plot", show_label=False)
                    gr.HTML('</div><br>')
                    
                    gr.HTML('<div class="panel-card">')
                    gr.Markdown("### 📋 Evaluator Safety Feedback")
                    report_md = gr.Markdown(value="*Evaluation report will be printed here.*")
                    gr.HTML('</div>')

        # ── TAB 2: 1v1 BATTLE ARENA ─────────────────
        with gr.Tab("⚔️ 1v1 Battle Arena"):
            gr.Markdown("### Compare driving parameters and declare a winner in a head-to-head match!")
            
            with gr.Row():
                # Driver 1 config
                with gr.Column(scale=1):
                    gr.HTML('<div class="panel-card" style="border: 1px solid #3b5bdb55 !important;">')
                    gr.Markdown("### 🚘 Driver A (Challenger 1)")
                    b_driver_name_1 = gr.Textbox(value="Challenger A", label="Driver Name")
                    b_video_1 = gr.Video(label="Driver A Clip", height=180)
                    b_preset_1 = gr.Dropdown(
                        choices=list(VEHICLE_PRESETS.keys()),
                        value=get_suggested_vehicle(),
                        label="Driver A Vehicle",
                    )
                    b_braking_1 = gr.Slider(minimum=2.0, maximum=8.0, value=3.5, step=0.1, label="Driver A Braking (s)")
                    gr.HTML('</div>')
                
                # Driver 2 config
                with gr.Column(scale=1):
                    gr.HTML('<div class="panel-card" style="border: 1px solid #7048e855 !important;">')
                    gr.Markdown("### 🚘 Driver B (Challenger 2)")
                    b_driver_name_2 = gr.Textbox(value="Challenger B", label="Driver Name")
                    b_video_2 = gr.Video(label="Driver B Clip", height=180)
                    b_preset_2 = gr.Dropdown(
                        choices=list(VEHICLE_PRESETS.keys()),
                        value=get_suggested_vehicle(),
                        label="Driver B Vehicle",
                    )
                    b_braking_2 = gr.Slider(minimum=2.0, maximum=8.0, value=3.5, step=0.1, label="Driver B Braking (s)")
                    gr.HTML('</div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<br><div class="panel-card">')
                    gr.Markdown("### ⚙️ Shared Environmental Settings")
                    b_speed_limit = gr.Slider(minimum=30, maximum=130, value=70, step=5, label="Shared Road Speed Limit (km/h)")
                    b_processing_speed = gr.Dropdown(
                        choices=["⚡ Fast (5 FPS)", "⚖️ Standard (10 FPS)", "🎯 High Precision (15 FPS)"],
                        value="⚖️ Standard (10 FPS)",
                        label="Shared Processing Speed & Quality",
                    )
                    gr.HTML('</div><br>')
                    
                    battle_btn = gr.Button(
                        "⚔️ START 1v1 BATTLE",
                        variant="primary",
                        elem_classes=["primary-btn"],
                    )
                
                with gr.Column(scale=2):
                    pass
            
            # Battle Results
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<br><div class="panel-card">')
                    gr.Markdown("### 🎬 Challenger A Output Stream")
                    b_video_out_1 = gr.Video(label="Driver A Annotated", height=240)
                    gr.HTML('</div>')
                with gr.Column(scale=1):
                    gr.HTML('<br><div class="panel-card">')
                    gr.Markdown("### 🎬 Challenger B Output Stream")
                    b_video_out_2 = gr.Video(label="Driver B Annotated", height=240)
                    gr.HTML('</div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<br><div class="panel-card">')
                    gr.Markdown("### 📊 Head-to-Head Comparison Card")
                    battle_html_out = gr.HTML(value="<div style='text-align:center;padding:40px;color:#444;font-family:monospace;'>Initiate a match to see the battle outcome.</div>")
                    gr.HTML('</div>')
                with gr.Column(scale=1):
                    gr.HTML('<br><div class="panel-card">')
                    gr.Markdown("### 📝 Combat Log Details")
                    battle_md_out = gr.Markdown(value="*Results will print here.*")
                    gr.HTML('</div>')

        # ── TAB 3: HISTORY & ANALYTICS ──────────────
        with gr.Tab("📊 Analytics & History Log"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML('<div class="panel-card">')
                    gr.Markdown("### 📋 Historical Evaluation Records")
                    history_table_out = gr.HTML(value=initial_history_table)
                    gr.HTML('</div>')
                
                with gr.Column(scale=1):
                    gr.HTML('<div class="panel-card">')
                    gr.Markdown("### 🛠️ History Actions")
                    clear_hist_btn = gr.Button("🗑️ CLEAR RUN LOGS")
                    gr.HTML('</div>')

        # ── TAB 4: VEHICLE PROFILES ─────────────────
        with gr.Tab("🔧 Vehicle Manager"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="panel-card">')
                    gr.Markdown("### ➕ Add Custom Vehicle Profile")
                    new_vehicle_name = gr.Textbox(placeholder="E.g., Honda Civic 2024", label="Vehicle Display Name")
                    new_braking_slider = gr.Slider(minimum=2.0, maximum=8.0, value=3.5, step=0.1, label="0-100 km/h braking time (s)")
                    save_vehicle_btn = gr.Button("💾 SAVE VEHICLE", variant="primary")
                    gr.HTML('</div>')
                
                with gr.Column(scale=2):
                    gr.HTML('<div class="panel-card">')
                    gr.Markdown("### 📖 Active Calibration Specifications")
                    gr.HTML("""
                    <div style="font-family:'Courier New',monospace;font-size:12px;color:#aaa;line-height:1.8;">
                      <b>Distance Calibration Model:</b> Pinhole Camera Model via Triangle Similarity<br>
                      <b>Physics Engine Constants:</b><br>
                      - Focal Length Constant: 800 px·m/px (Calibrated for 1080p lens at ~60° HFOV)<br>
                      - Reaction Leniency Multiplier: 1.2x (Pakistani traffic norms correction factor)<br>
                      - Human Reaction Speed: 1.5 seconds default safe reaction buffer<br><br>
                      <b>Representative Object Width Standards:</b><br>
                      - Pedestrians: 0.55m  |  Motorcycles: 0.80m  |  Bicycles: 0.65m<br>
                      - Standard Hatchbacks/Sedans: 1.60m - 1.80m<br>
                      - Buses and Heavy Trucks: 2.50m
                    </div>
                    """)
                    gr.HTML('</div>')

    # ── Wire preset dropdowns to update sliders ──
    vehicle_preset.change(
        fn=update_braking_from_preset,
        inputs=[vehicle_preset],
        outputs=[braking_slider, vehicle_name_box],
    )
    
    b_preset_1.change(
        fn=update_braking_from_preset,
        inputs=[b_preset_1],
        outputs=[b_braking_1, gr.State()], # Throwaway name
    )
    
    b_preset_2.change(
        fn=update_braking_from_preset,
        inputs=[b_preset_2],
        outputs=[b_braking_2, gr.State()],
    )

    # ── Wire Evaluate Button ─────────────────────
    # Outputs: video_output, report_md, score_html, telemetry_chart, vehicle_preset choice, history_table_out
    analyse_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            vehicle_name_box,
            braking_slider,
            speed_limit_slider,
            processing_speed_dropdown,
        ],
        outputs=[
            video_output,
            report_md,
            score_html,
            telemetry_chart,
            vehicle_preset, # Update dropdown choices in both tabs
            history_table_out,
        ],
    )

    # ── Wire 1v1 Battle Button ───────────────────
    battle_btn.click(
        fn=process_1v1_battle,
        inputs=[
            b_video_1, b_driver_name_1, b_preset_1, b_braking_1,
            b_video_2, b_driver_name_2, b_preset_2, b_braking_2,
            b_speed_limit,
            b_processing_speed,
        ],
        outputs=[
            b_video_out_1,
            b_video_out_2,
            battle_html_out,
            battle_md_out,
            vehicle_preset,
            history_table_out
        ],
    )

    # ── Wire Vehicle Profiles Manager ────────────
    save_vehicle_btn.click(
        fn=add_new_vehicle_action,
        inputs=[new_vehicle_name, new_braking_slider],
        outputs=[vehicle_preset, report_md], # Shows success message in feedback box
    )
    
    # Reload preset lists when save button clicked
    def refresh_dropdowns():
        presets = build_vehicle_presets()
        return gr.update(choices=list(presets.keys())), gr.update(choices=list(presets.keys()))
        
    save_vehicle_btn.click(
        fn=refresh_dropdowns,
        outputs=[b_preset_1, b_preset_2]
    )

    # ── Wire Clear History Actions ───────────────
    clear_hist_btn.click(
        fn=clear_history_action,
        outputs=[vehicle_preset, history_table_out], # Updates history dropdown and history table
    )

    # ── Footer ────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;padding:30px 0 10px;
         font-family:'Courier New',monospace;font-size:11px;
         color:#4a4a6d;letter-spacing:0.12em;">
      DRIVESENSE AI · SOFTWARE ENGINEERING SEMESTER SUBMISSION · ITU LAHORE BSAI
    </div>
    """)


#  ENTRY POINT
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=CSS,
    )