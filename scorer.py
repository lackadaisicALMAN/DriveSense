"""
scorer.py  –  DriveSense AI  |  Safety Scoring & Report Engine
===============================================================
Converts a list of FrameEvent objects into:
  • A numeric safety score  (0.0 – 10.0)
  • A structured report dict with per-category breakdowns
  • A human-readable markdown report string
  • Personalised improvement suggestions
"""

from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass
from detector import FrameEvent, DriverProfile


# ─────────────────────────────────────────────
#  SCORING WEIGHTS  (must sum to 1.0)
# ─────────────────────────────────────────────
WEIGHTS = {
    "following_distance" : 0.40,   # most safety-critical
    "lane_discipline"    : 0.35,
    "traffic_light"      : 0.25,
}

# How much the score drops per infraction (out of 10)
PENALTIES = {
    "too_close_per_frame"    : 0.15,   # per frame flagged (after normalising)
    "lane_drift_per_event"   : 0.8,    # per continuous drift event
    "lane_over_per_event"    : 1.2,    # per over-line event (tyre on line)
    "light_violation"        : 2.5,    # per green-light stationary incident
}


# ─────────────────────────────────────────────
#  REPORT DATACLASS
# ─────────────────────────────────────────────
@dataclass
class SafetyReport:
    overall_score      : float
    distance_score     : float
    lane_score         : float
    light_score        : float

    total_frames       : int
    analysed_frames    : int
    duration_s         : float

    too_close_count    : int
    drift_events       : int
    over_line_events   : int
    light_violations   : int
    lights_seen        : int

    vehicle_name       : str

    suggestions        : List[str]
    markdown           : str


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

        # ── 1. Following Distance ──────────────────
        too_close_frames = sum(1 for e in events if e.too_close)
        # As a fraction of frames where a car was visible
        car_visible = sum(1 for e in events if e.nearest_car_dist_m is not None)
        if car_visible > 0:
            close_frac  = too_close_frames / car_visible
            # Exponential penalty: the more frames too close, the harsher
            dist_penalty = min(10.0, close_frac * 18.0)
        else:
            dist_penalty = 0.0
        distance_score = max(0.0, 10.0 - dist_penalty)

        # ── 2. Lane Discipline ─────────────────────
        drift_events, over_line_events = self._count_lane_events(events)
        lane_penalty = (drift_events    * PENALTIES["lane_drift_per_event"] +
                        over_line_events * PENALTIES["lane_over_per_event"])
        lane_score   = max(0.0, 10.0 - lane_penalty)

        # ── 3. Traffic Light ──────────────────────
        light_violations = sum(1 for e in events if e.light_violation)
        lights_seen      = sum(1 for e in events if e.light_detected)
        light_penalty    = light_violations * PENALTIES["light_violation"]
        light_score      = max(0.0, 10.0 - light_penalty)

        # ── 4. Weighted Overall ────────────────────
        overall = (
            WEIGHTS["following_distance"] * distance_score +
            WEIGHTS["lane_discipline"]    * lane_score     +
            WEIGHTS["traffic_light"]      * light_score
        )
        overall = round(min(10.0, max(0.0, overall)), 1)

        # ── 5. Suggestions ────────────────────────
        suggestions = self._build_suggestions(
            distance_score, lane_score, light_score,
            too_close_frames, drift_events, over_line_events,
            light_violations, car_visible
        )

        # ── 6. Markdown report ────────────────────
        md = self._build_markdown(
            overall, distance_score, lane_score, light_score,
            n, duration_s, too_close_frames, drift_events,
            over_line_events, light_violations, lights_seen,
            suggestions
        )

        return SafetyReport(
            overall_score    = overall,
            distance_score   = round(distance_score, 1),
            lane_score       = round(lane_score, 1),
            light_score      = round(light_score, 1),
            total_frames     = n,
            analysed_frames  = n,
            duration_s       = round(duration_s, 1),
            too_close_count  = too_close_frames,
            drift_events     = drift_events,
            over_line_events = over_line_events,
            light_violations = light_violations,
            lights_seen      = lights_seen,
            vehicle_name     = self.profile.vehicle_name,
            suggestions      = suggestions,
            markdown         = md,
        )

    # ──────────────────────────────────────────
    #  COUNT LANE EVENTS  (run-length encoding)
    # ──────────────────────────────────────────
    @staticmethod
    def _count_lane_events(events: List[FrameEvent]):
        """
        Count distinct continuous DRIFT / OVER_LINE events.
        A new event starts when the status changes from OK.
        """
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
    #  SUGGESTIONS
    # ──────────────────────────────────────────
    def _build_suggestions(
        self, dist_sc, lane_sc, light_sc,
        too_close, drifts, over_lines, light_viols, car_visible
    ) -> List[str]:
        tips = []

        if dist_sc < 7.0:
            safe_d = self.profile.safe_distance_m
            tips.append(
                f"⚠️  **Following Distance**: You were too close to the vehicle ahead "
                f"in {too_close} frames. At {self.profile.speed_limit_kmh:.0f} km/h, "
                f"maintain at least **{safe_d:.0f} m** of gap. "
                f"Remember: your {self.profile.vehicle_name} needs "
                f"~{self.profile.braking_100_sec:.1f}s to stop from 100 km/h."
            )
        elif dist_sc < 9.0:
            tips.append(
                f"✅  Following distance was mostly good. Keep the "
                f"{self.profile.safe_distance_m:.0f} m buffer as a habit."
            )

        if over_lines > 0:
            tips.append(
                f"⚠️  **Lane Discipline – Over the Line**: You straddled or crossed "
                f"lane markings {over_lines} time(s). Keep your tyres well within the "
                f"lane boundaries and use cat's-eye reflectors as a guide at night."
            )
        if drifts > 0:
            tips.append(
                f"⚠️  **Lane Drift**: The system detected {drifts} drift event(s). "
                f"Check mirrors more frequently and keep both hands on the wheel to "
                f"maintain a straight path."
            )
        if lane_sc >= 9.5:
            tips.append("✅  Excellent lane discipline throughout the clip.")

        if light_viols > 0:
            tips.append(
                f"⚠️  **Traffic Light Response**: You failed to respond to a green "
                f"signal {light_viols} time(s) (stationary for >2 seconds with no "
                f"vehicle ahead). Stay alert at junctions — scan signals early."
            )
        elif light_sc == 10.0:
            tips.append("✅  Good signal awareness — no green-light hesitation detected.")

        if not tips:
            tips.append("🏆  Outstanding drive! No significant issues detected.")

        return tips

    # ──────────────────────────────────────────
    #  MARKDOWN REPORT
    # ──────────────────────────────────────────
    def _build_markdown(
        self, overall, dist_sc, lane_sc, light_sc,
        frames, duration, too_close, drifts, over_lines,
        light_viols, lights_seen, suggestions
    ) -> str:

        def stars(s):
            filled = round(s / 2)
            return "★" * filled + "☆" * (5 - filled)

        grade_map = [(9, "A+"), (8, "A"), (7, "B"), (6, "C"), (5, "D"), (0, "F")]
        grade = next(g for threshold, g in grade_map if overall >= threshold)

        lines = [
            f"# 🚗 DriveSense AI — Safety Report",
            f"",
            f"**Vehicle:** {self.profile.vehicle_name}  |  "
            f"**Clip duration:** {duration:.1f}s  |  "
            f"**Frames analysed:** {frames}",
            f"",
            f"---",
            f"",
            f"## Overall Score:  {overall} / 10   ({grade})",
            f"",
            f"| Category | Score | Stars |",
            f"|----------|-------|-------|",
            f"| 🚘 Following Distance | {dist_sc:.1f} / 10 | {stars(dist_sc)} |",
            f"| 🛣️  Lane Discipline    | {lane_sc:.1f} / 10 | {stars(lane_sc)} |",
            f"| 🚦 Traffic Lights     | {light_sc:.1f} / 10 | {stars(light_sc)} |",
            f"",
            f"---",
            f"",
            f"## Incident Summary",
            f"",
            f"- **Too-close frames:** {too_close}",
            f"- **Lane drift events:** {drifts}",
            f"- **Over-line events:** {over_lines}",
            f"- **Green-light violations:** {light_viols}",
            f"- **Traffic lights detected:** {lights_seen}",
            f"",
            f"---",
            f"",
            f"## Personalised Feedback",
            f"",
        ]
        for tip in suggestions:
            lines.append(f"- {tip}")
        lines += [
            f"",
            f"---",
            f"",
            f"*Generated by DriveSense AI — for educational purposes only.*",
        ]
        return "\n".join(lines)

    # ──────────────────────────────────────────
    def _empty_report(self) -> SafetyReport:
        return SafetyReport(
            overall_score=0.0, distance_score=0.0, lane_score=0.0,
            light_score=0.0, total_frames=0, analysed_frames=0, duration_s=0.0,
            too_close_count=0, drift_events=0, over_line_events=0,
            light_violations=0, lights_seen=0,
            vehicle_name=self.profile.vehicle_name,
            suggestions=["No frames were analysed. Please upload a valid video."],
            markdown="# Error\n\nNo frames were analysed.",
        )
