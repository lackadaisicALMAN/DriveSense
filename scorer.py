"""
scorer.py  –  DriveSense AI  |  Safety Scoring & Report Engine
===============================================================
Converts a list of FrameEvent objects into:
  • A numeric safety score  (0.0 – 10.0)
  • A structured report dict with per-category breakdowns
  • A human-readable markdown report string
  • Personalised improvement suggestions
  • A chronological timeline of driving incidents
"""

from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass, field
from detector import FrameEvent, DriverProfile


# ─────────────────────────────────────────────
#  SCORING WEIGHTS  (must sum to 1.0)
# ─────────────────────────────────────────────
WEIGHTS = {
    "following_distance" : 0.40,   # following gap & forward collision risk
    "lane_discipline"    : 0.35,   # lane drift & over-line infractions
    "intersection_safety": 0.25,   # traffic signals & stop signs
}

# How much the score drops per infraction (out of 10)
PENALTIES = {
    "too_close_per_frame"    : 0.15,   # tailgating
    "lane_drift_per_event"   : 0.8,    # lane drift
    "lane_over_per_event"    : 1.2,    # wheels over line
    "light_violation"        : 2.5,    # green-light stationary delay
    "stop_sign_violation"    : 2.0,    # rolling through stop sign
    "collision_warning_event": 1.5,    # critical TTC event
}


# ─────────────────────────────────────────────
#  REPORT DATACLASS
# ─────────────────────────────────────────────
@dataclass
class SafetyReport:
    overall_score       : float
    distance_score      : float
    lane_score          : float
    light_score         : float  # Intersection safety (retained name for UI compatibility)

    total_frames        : int
    analysed_frames     : int
    duration_s          : float

    too_close_count     : int
    drift_events        : int
    over_line_events    : int
    light_violations    : int
    lights_seen         : int

    vehicle_name        : str
    suggestions         : List[str]
    markdown            : str
    
    stop_sign_violations: int = 0
    stop_signs_seen     : int = 0
    collision_warnings  : int = 0
    timeline            : List[Dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────
#  SCORER
# ─────────────────────────────────────────────
class SafetyScorer:

    def __init__(self, profile: DriverProfile):
        self.profile = profile

    def score(self, events: List[FrameEvent]) -> SafetyReport:
        if not events:
            return self._empty_report()

        n = len(events)
        duration_s = events[-1].timestamp_s

        # ── 1. Following Distance & Collision Warnings ──────────────────
        too_close_frames = sum(1 for e in events if e.too_close)
        car_visible = sum(1 for e in events if e.nearest_car_dist_m is not None)
        
        if car_visible > 0:
            close_frac  = too_close_frames / car_visible
            dist_penalty = min(8.0, close_frac * 15.0)  # max 8 points penalty for tailgating
        else:
            dist_penalty = 0.0
            
        collision_events = self._count_collision_events(events)
        collision_penalty = min(5.0, collision_events * PENALTIES["collision_warning_event"])
        
        distance_score = max(0.0, 10.0 - dist_penalty - collision_penalty)

        # ── 2. Lane Discipline ─────────────────────
        drift_events, over_line_events = self._count_lane_events(events)
        lane_penalty = (drift_events    * PENALTIES["lane_drift_per_event"] +
                        over_line_events * PENALTIES["lane_over_per_event"])
        lane_score   = max(0.0, 10.0 - lane_penalty)

        # ── 3. Intersection Safety (Traffic Lights & Stop Signs) ──────
        light_violations = sum(1 for e in events if e.light_violation)
        lights_seen      = sum(1 for e in events if e.light_detected)
        
        stop_sign_violations = sum(1 for e in events if e.stop_sign_violation)
        stop_signs_seen      = sum(1 for e in events if e.stop_sign_detected)
        
        intersection_penalty = (light_violations * PENALTIES["light_violation"] +
                                stop_sign_violations * PENALTIES["stop_sign_violation"])
        intersection_score   = max(0.0, 10.0 - intersection_penalty)

        # ── 4. Weighted Overall ────────────────────
        overall = (
            WEIGHTS["following_distance"] * distance_score +
            WEIGHTS["lane_discipline"]    * lane_score     +
            WEIGHTS["intersection_safety"] * intersection_score
        )
        overall = round(min(10.0, max(0.0, overall)), 1)

        # ── 5. Build chronological incident timeline ──
        timeline = self._build_timeline(events)

        # ── 6. Suggestions ────────────────────────
        suggestions = self._build_suggestions(
            distance_score, lane_score, intersection_score,
            too_close_frames, drift_events, over_line_events,
            light_violations, stop_sign_violations, collision_events
        )

        # ── 7. Markdown report ────────────────────
        md = self._build_markdown(
            overall, distance_score, lane_score, intersection_score,
            n, duration_s, too_close_frames, drift_events,
            over_line_events, light_violations, stop_sign_violations,
            collision_events, lights_seen, stop_signs_seen, suggestions, timeline
        )

        return SafetyReport(
            overall_score    = overall,
            distance_score   = round(distance_score, 1),
            lane_score       = round(lane_score, 1),
            light_score      = round(intersection_score, 1), # mapped to light_score for UI
            total_frames     = n,
            analysed_frames  = n,
            duration_s       = round(duration_s, 1),
            too_close_count  = too_close_frames,
            drift_events     = drift_events,
            over_line_events = over_line_events,
            light_violations = light_violations,
            lights_seen      = lights_seen,
            stop_sign_violations = stop_sign_violations,
            stop_signs_seen  = stop_signs_seen,
            collision_warnings = collision_events,
            timeline         = timeline,
            vehicle_name     = self.profile.vehicle_name,
            suggestions      = suggestions,
            markdown         = md,
        )

    # ──────────────────────────────────────────
    #  COUNT LANE EVENTS
    # ──────────────────────────────────────────
    @staticmethod
    def _count_lane_events(events: List[FrameEvent]):
        drift_events = over_line_events = 0
        prev = "OK"
        for e in events:
            s = e.lane_status
            if s == "DRIFT"     and prev != "DRIFT":
                drift_events += 1
            elif s == "OVER_LINE" and prev != "OVER_LINE":
                over_line_events += 1
            prev = s
        return drift_events, over_line_events

    # ──────────────────────────────────────────
    #  COUNT COLLISION WARNINGS
    # ──────────────────────────────────────────
    @staticmethod
    def _count_collision_events(events: List[FrameEvent]) -> int:
        count = 0
        prev = False
        for e in events:
            curr = e.collision_warning
            if curr and not prev:
                count += 1
            prev = curr
        return count

    # ──────────────────────────────────────────
    #  BUILD CHRONOLOGICAL TIMELINE
    # ──────────────────────────────────────────
    def _build_timeline(self, events: List[FrameEvent]) -> List[Dict[str, Any]]:
        timeline = []
        
        in_tailgate = False
        tailgate_start_ts = 0.0
        
        in_fcw = False
        fcw_start_ts = 0.0
        
        in_drift = False
        drift_start_ts = 0.0
        
        in_over = False
        over_start_ts = 0.0
        
        for e in events:
            ts = e.timestamp_s
            time_str = f"{int(ts // 60):02d}:{int(ts % 60):02d}"
            
            # Tailgating
            if e.too_close:
                if not in_tailgate:
                    in_tailgate = True
                    tailgate_start_ts = ts
            else:
                if in_tailgate:
                    duration = ts - tailgate_start_ts
                    if duration > 1.0:
                        timeline.append({
                            "timestamp": tailgate_start_ts,
                            "time_str": f"{int(tailgate_start_ts // 60):02d}:{int(tailgate_start_ts % 60):02d}",
                            "type": "⚠️ Tailgating",
                            "severity": "Warning",
                            "message": f"Tailgating vehicle (gap fell to {e.nearest_car_dist_m or 0:.1f}m, safe buffer is {self.profile.safe_distance_m:.1f}m)."
                        })
                    in_tailgate = False
                    
            # Collision Warning (FCW)
            if e.collision_warning:
                if not in_fcw:
                    in_fcw = True
                    fcw_start_ts = ts
            else:
                if in_fcw:
                    timeline.append({
                        "timestamp": fcw_start_ts,
                        "time_str": f"{int(fcw_start_ts // 60):02d}:{int(fcw_start_ts % 60):02d}",
                        "type": "🚨 Collision Alert",
                        "severity": "Danger",
                        "message": f"Forward Collision Warning! Braking time-to-collision (TTC) dropped to {e.ttc_s or 0.0:.1f}s."
                    })
                    in_fcw = False
                    
            # Lane Drift
            if e.lane_status == "DRIFT":
                if not in_drift:
                    in_drift = True
                    drift_start_ts = ts
            else:
                if in_drift:
                    timeline.append({
                        "timestamp": drift_start_ts,
                        "time_str": f"{int(drift_start_ts // 60):02d}:{int(drift_start_ts % 60):02d}",
                        "type": "🛣️ Lane Drift",
                        "severity": "Warning",
                        "message": "Vehicle drifted from the center path without lane markings on both sides."
                    })
                    in_drift = False
                    
            # Over Lane
            if e.lane_status == "OVER_LINE":
                if not in_over:
                    in_over = True
                    over_start_ts = ts
            else:
                if in_over:
                    timeline.append({
                        "timestamp": over_start_ts,
                        "time_str": f"{int(over_start_ts // 60):02d}:{int(over_start_ts % 60):02d}",
                        "type": "⛔ Lane straddle",
                        "severity": "Danger",
                        "message": "Car tyres crossed the lane boundaries. Active lane departure warning."
                    })
                    in_over = False
                    
            # Traffic Light Violation
            if e.light_violation:
                last_violations = [t for t in timeline if t["type"] == "🚦 Traffic Signal Delay"]
                if not last_violations or (ts - last_violations[-1]["timestamp"] > 5.0):
                    timeline.append({
                        "timestamp": ts,
                        "time_str": time_str,
                        "type": "🚦 Traffic Signal Delay",
                        "severity": "Danger",
                        "message": "Failed to proceed at a green traffic light (stationary for > 2 seconds)."
                    })
                    
            # Stop Sign Violation
            if e.stop_sign_violation:
                last_violations = [t for t in timeline if t["type"] == "🛑 Stop Sign Roll"]
                if not last_violations or (ts - last_violations[-1]["timestamp"] > 5.0):
                    timeline.append({
                        "timestamp": ts,
                        "time_str": time_str,
                        "type": "🛑 Stop Sign Roll",
                        "severity": "Danger",
                        "message": "Failed to stop at a stop sign (rolled past without stopping)."
                    })
                    
        # Flush remaining active states
        if in_tailgate:
            timeline.append({
                "timestamp": tailgate_start_ts,
                "time_str": f"{int(tailgate_start_ts // 60):02d}:{int(tailgate_start_ts % 60):02d}",
                "type": "⚠️ Tailgating",
                "severity": "Warning",
                "message": "Tailgating vehicle ahead at the end of the video."
            })
        if in_fcw:
            timeline.append({
                "timestamp": fcw_start_ts,
                "time_str": f"{int(fcw_start_ts // 60):02d}:{int(fcw_start_ts % 60):02d}",
                "type": "🚨 Collision Alert",
                "severity": "Danger",
                "message": "Active collision risk at the end of the video."
            })
        if in_drift:
            timeline.append({
                "timestamp": drift_start_ts,
                "time_str": f"{int(drift_start_ts // 60):02d}:{int(drift_start_ts % 60):02d}",
                "type": "🛣️ Lane Drift",
                "severity": "Warning",
                "message": "Vehicle drifting at the end of the video."
            })
        if in_over:
            timeline.append({
                "timestamp": over_start_ts,
                "time_str": f"{int(over_start_ts // 60):02d}:{int(over_start_ts % 60):02d}",
                "type": "⛔ Lane straddle",
                "severity": "Danger",
                "message": "Vehicle straddling lane line at the end of the video."
            })
            
        timeline.sort(key=lambda x: x["timestamp"])
        return timeline

    # ──────────────────────────────────────────
    #  SUGGESTIONS
    # ──────────────────────────────────────────
    def _build_suggestions(
        self, dist_sc, lane_sc, inter_sc,
        too_close, drifts, over_lines, light_viols, stop_viols, collision_events
    ) -> List[str]:
        tips = []

        if dist_sc < 7.0:
            safe_d = self.profile.safe_distance_m
            tips.append(
                f"⚠️ **Following Gap**: Keep a larger safety buffer. At {self.profile.speed_limit_kmh:.0f} km/h, "
                f"your {self.profile.vehicle_name} needs **{safe_d:.0f} m** to stop in an emergency. "
                f"Apply the '3-Second Rule' to estimate gaps on the highway."
            )
        if collision_events > 0:
            tips.append(
                f"🚨 **Collision Risk**: The system detected {collision_events} critical closing rate events. "
                f"Look further down the road to anticipate decelerations early rather than reacting at the last second."
            )
        elif dist_sc >= 9.0:
            tips.append("✅ **Following Distance**: Good. Kept safe margins from obstacles.")

        if over_lines > 0:
            tips.append(
                f"🛣️ **Lane Markings**: You straddled lane boundaries {over_lines} times. "
                f"Make sure to use indicators before changing lanes, and center yourself between lane markings."
            )
        if drifts > 0:
            tips.append(
                f"⚠️ **Steering Control**: System detected {drifts} drift events. Keep two hands on the wheel "
                f"and avoid driving distractions (mobile phones, dashboard screens)."
            )
        if lane_sc >= 9.5:
            tips.append("✅ **Lane Keeping**: Excellent lane discipline and stability.")

        if light_viols > 0:
            tips.append(
                f"🚦 **Signal Awareness**: Delayed reaction at green light ({light_viols} times). "
                f"Pay active attention at intersections to keep traffic flowing safely."
            )
        if stop_viols > 0:
            tips.append(
                f"🛑 **Stop Signs**: Rolled through {stop_viols} stop sign junctions. "
                f"You must come to a complete stop (0 km/h) at a stop line, scan the intersection, and then proceed."
            )
        if inter_sc == 10.0:
            tips.append("✅ **Intersection Safety**: Perfect response to traffic signals and stop lines.")

        return tips

    # ──────────────────────────────────────────
    #  MARKDOWN REPORT GENERATOR
    # ──────────────────────────────────────────
    def _build_markdown(
        self, overall, dist_sc, lane_sc, inter_sc,
        frames, duration, too_close, drifts, over_lines,
        light_viols, stop_viols, collision_events, lights_seen, stop_signs_seen,
        suggestions, timeline
    ) -> str:

        def stars(s):
            filled = round(s / 2)
            return "★" * filled + "☆" * (5 - filled)

        grade_map = [(9, "A+"), (8, "A"), (7, "B"), (6, "C"), (5, "D"), (0, "F")]
        grade = next(g for threshold, g in grade_map if overall >= threshold)

        lines = [
            f"# 🚗 DriveSense AI — Safety Scorecard",
            f"",
            f"**Vehicle Profile:** {self.profile.vehicle_name}  |  "
            f"**Clip Duration:** {duration:.1f}s  |  "
            f"**Analysed Frames:** {frames}",
            f"",
            f"---",
            f"",
            f"## Overall Score:  **{overall} / 10**   ({grade})",
            f"",
            f"| Driving Category | Score | Star Rating |",
            f"| :--- | :--- | :--- |",
            f"| 🚘 Following Distance & TTC | {dist_sc:.1f} / 10 | {stars(dist_sc)} |",
            f"| 🛣️ Lane Keeping & Discipline | {lane_sc:.1f} / 10 | {stars(lane_sc)} |",
            f"| 🚦 Intersection Safety | {inter_sc:.1f} / 10 | {stars(inter_sc)} |",
            f"",
            f"---",
            f"",
            f"## Incidents & Telemetry Summary",
            f"",
            f"- **Unsafe Following Frames:** {too_close} frames",
            f"- **Collision Warning Events (TTC < 2s):** {collision_events} warnings",
            f"- **Lane Drifts:** {drifts} times",
            f"- **Lane Marker Straddling:** {over_lines} times",
            f"- **Green Light Delay Incidents:** {light_viols} times",
            f"- **Stop Sign Violations:** {stop_viols} times",
            f"",
            f"---",
            f"",
            f"## 📋 Timeline of Driving Incidents",
            f""
        ]
        
        if timeline:
            lines.append("| Time | Incident Type | Description | Severity |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for item in timeline:
                sev_badge = f"<span style='color:#ff5555;font-weight:bold;'>{item['severity']}</span>" if item['severity'] == "Danger" else f"<span style='color:#ffaa00;font-weight:bold;'>{item['severity']}</span>"
                lines.append(f"| **{item['time_str']}** | {item['type']} | {item['message']} | {sev_badge} |")
        else:
            lines.append("🏆 *No critical incidents detected! Perfect driving log.*")
            
        lines.extend([
            f"",
            f"---",
            f"",
            f"## 💡 Personalised Feedback & Suggestions",
            f"",
        ])
        for tip in suggestions:
            lines.append(f"- {tip}")
            
        lines.extend([
            f"",
            f"---",
            f"",
            f"*Generated by DriveSense AI (Semester Project Submission).*",
        ])
        return "\n".join(lines)

    # ──────────────────────────────────────────
    #  EMPTY FALLBACK
    # ──────────────────────────────────────────
    def _empty_report(self) -> SafetyReport:
        return SafetyReport(
            overall_score=0.0, distance_score=0.0, lane_score=0.0,
            light_score=0.0, total_frames=0, analysed_frames=0, duration_s=0.0,
            too_close_count=0, drift_events=0, over_line_events=0,
            light_violations=0, lights_seen=0, stop_sign_violations=0, stop_signs_seen=0,
            collision_warnings=0, timeline=[],
            vehicle_name=self.profile.vehicle_name,
            suggestions=["No frames were analysed. Please upload a valid video."],
            markdown="# Error\n\nNo frames were analysed.",
        )
