"""
detector.py  –  DriveSense AI  |  Core Detection Engine
=========================================================
Handles:
  • Car / truck detection with own-car masking
  • Real-world distance estimation from bounding-box geometry
  • Lane-line detection via Hough transforms + perspective cues
  • Traffic-light detection + colour classification (R/G/Y)
  • Per-frame event logging for the scorer
"""

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Optional
import math

# ─────────────────────────────────────────────
#  CONSTANTS  (tunable via DriverProfile)
# ─────────────────────────────────────────────
COCO_CAR_IDS    = {2, 5, 7}   # car, bus, truck
COCO_LIGHT_ID   = 9            # traffic light

# Approximate focal-length calibration constant (pixels × metres / pixels).
# Calibrated for a typical dashcam with ~1920 px wide sensor and ~60° HFOV.
# Used in triangle similarity: distance = (FOCAL_CONST * vehicle_width_m) / pixel_width
FOCAL_CONST     = 800          # f * known_width_m  (tune per camera if needed)

# Own-car mask: ignore detections in the bottom-centre strip
# (bonnet / dashboard visible in many dashcams).
OWN_CAR_MASK_FRAC_Y  = 0.82   # ignore detections whose bottom > this × height
OWN_CAR_MASK_FRAC_X  = (0.25, 0.75)  # and whose centre-x is in this fraction range

# Traffic light: minimum confidence to classify colour
LIGHT_MIN_CONF = 0.40

# ─────────────────────────────────────────────
#  COMPREHENSIVE VEHICLE DATABASE  (PakWheels / OEM specs)
#  Triangle Similarity Distance Calculation:
#    depth = (FOCAL_CONST * real_width_m) / bounding_box_width_px
#  This is far more robust than Y-coordinate methods on hills/curves
# ─────────────────────────────────────────────
VEHICLE_DATABASE = {
    "Suzuki Mehran": {
        "width_m": 1.61,
        "length_m": 3.8,
        "front_m": 1.8,
    },
    "Suzuki Alto": {
        "width_m": 1.62,
        "length_m": 3.86,
        "front_m": 1.7,
    },
    "Suzuki Wagon R": {
        "width_m": 1.68,
        "length_m": 4.17,
        "front_m": 1.8,
    },
    "Toyota Corolla": {
        "width_m": 1.80,
        "length_m": 4.63,
        "front_m": 1.9,
    },
    "Honda Civic": {
        "width_m": 1.80,
        "length_m": 4.63,
        "front_m": 2.0,
    },
    "Honda City": {
        "width_m": 1.68,
        "length_m": 4.42,
        "front_m": 1.9,
    },
    "Toyota Yaris": {
        "width_m": 1.70,
        "length_m": 4.43,
        "front_m": 1.8,
    },
    "Toyota Hilux": {
        "width_m": 1.86,
        "length_m": 5.35,
        "front_m": 2.5,
    },
    "Suzuki Jimny": {
        "width_m": 1.64,
        "length_m": 3.64,
        "front_m": 1.9,
    },
    "Toyota Prius": {
        "width_m": 1.77,
        "length_m": 4.63,
        "front_m": 2.1,
    },
    "Honda N-WGN": {
        "width_m": 1.68,
        "length_m": 3.88,
        "front_m": 1.85,
    },
    "Daihatsu Mira": {
        "width_m": 1.60,
        "length_m": 3.70,
        "front_m": 1.6,
    },
    "Hyundai i10": {
        "width_m": 1.68,
        "length_m": 3.85,
        "front_m": 1.7,
    },
    "KIA Picanto": {
        "width_m": 1.63,
        "length_m": 3.84,
        "front_m": 1.75,
    },
    "Toyota Fortuner": {
        "width_m": 1.86,
        "length_m": 4.83,
        "front_m": 2.2,
    },
    "Chevrolet Bolan": {
        "width_m": 1.68,
        "length_m": 4.25,
        "front_m": 1.95,
    },
    "General Truck": {
        "width_m": 2.50,
        "length_m": 6.0,
        "front_m": 2.8,
    },
    "General Bus": {
        "width_m": 2.50,
        "length_m": 10.0,
        "front_m": 2.5,
    },
    "Custom Vehicle": {
        "width_m": 1.75,
        "length_m": 4.5,
        "front_m": 1.85,
    },
    "Standard Car": {
        "width_m": 1.75,
        "length_m": 4.5,
        "front_m": 1.85,
    },
}

# Backward compatibility: extract front lengths only
VEHICLE_FRONT_LENGTH_M = {k: v["front_m"] for k, v in VEHICLE_DATABASE.items()}

# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────
@dataclass
class DriverProfile:
    """User-supplied vehicle / braking profile."""
    vehicle_name    : str   = "Standard Car"
    braking_100_sec : float = 3.5    # seconds to stop from 100 km/h
    speed_limit_kmh : float = 70.0   # assumed road speed limit

    @property
    def vehicle_spec(self) -> dict:
        """
        Return full vehicle specification (width, length, front offset).
        Falls back to 'Standard Car' if vehicle not in database.
        """
        return VEHICLE_DATABASE.get(self.vehicle_name, VEHICLE_DATABASE["Standard Car"])

    @property
    def vehicle_width_m(self) -> float:
        """Vehicle width in metres (used for triangle similarity distance)."""
        return self.vehicle_spec.get("width_m", 1.75)

    @property
    def vehicle_front_length_m(self) -> float:
        """Camera-to-bumper distance (used when interior is visible)."""
        return self.vehicle_spec.get("front_m", 1.85)

    @property
    def safe_distance_m(self) -> float:
        """
        Safe following distance = reaction_distance + braking_distance.
        Uses the 2-second rule plus the vehicle's own braking distance at speed_limit.
        Formula: d = v × t_reaction + v²/(2a)
        v in m/s, a from braking profile.
        """
        v = self.speed_limit_kmh / 3.6         # m/s
        t_reaction = 1.5                        # s (human reaction)
        # deceleration from braking profile: v0/t  (0→100 reversed)
        v100 = 100 / 3.6
        a = v100 / max(self.braking_100_sec, 0.5)
        d_braking = (v ** 2) / (2 * a)
        d_reaction = v * t_reaction
        # Apply a 1.2× leniency factor as per SRS
        return (d_reaction + d_braking) * 1.2


@dataclass
class FrameEvent:
    """One frame's analysis result."""
    frame_idx       : int
    timestamp_s     : float
    # Car following
    nearest_car_dist_m : Optional[float] = None    # adjusted distance (includes vehicle length if interior visible)
    raw_nearest_dist_m : Optional[float] = None    # raw bounding-box distance
    interior_visible : bool = False                 # True if steering/meter/dashboard detected
    too_close       : bool = False
    # Lane
    lane_status     : str  = "OK"        # OK | DRIFT | OVER_LINE
    # Traffic light
    light_detected  : bool = False
    light_color     : str  = "NONE"      # RED | GREEN | YELLOW | UNKNOWN
    light_violation : bool = False       # car stationary at green too long
    # Annotated frame (written by annotate())
    annotated_frame : Optional[np.ndarray] = None


# ─────────────────────────────────────────────
#  DETECTOR CLASS
# ─────────────────────────────────────────────
class DriveSenseDetector:

    def __init__(self, profile: DriverProfile, model_name: str = "yolo11n.pt"):
        print(f"[DriveSense] Loading model: {model_name}")
        self.model   = YOLO(model_name)
        self.profile = profile

        # State for traffic-light green-stationary timer
        self._green_stationary_frames = 0
        self._GREEN_FRAMES_THRESHOLD  = 0  # set per fps in process_video

        # Previous frame for optical flow (crude motion detection)
        self._prev_gray : Optional[np.ndarray] = None

        # Interior visibility detection (rolling average)
        self._interior_visible_count = 0
        self._interior_detection_window = 5  # frames to track

    # ──────────────────────────────────────────
    #  MAIN PER-FRAME ENTRY POINT
    # ──────────────────────────────────────────
    def analyse_frame(self, frame: np.ndarray, frame_idx: int,
                      fps: float, green_frames_threshold: int) -> FrameEvent:
        h, w = frame.shape[:2]
        ts   = frame_idx / max(fps, 1)
        event = FrameEvent(frame_idx=frame_idx, timestamp_s=ts)

        # ── 1. YOLO inference ──
        results = self.model(frame, verbose=False, conf=0.35)
        boxes   = results[0].boxes if results else []

        car_boxes   = []
        light_boxes = []

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            xyxy   = box.xyxy[0].cpu().numpy().astype(int)

            if cls_id in COCO_CAR_IDS:
                if not self._is_own_car(xyxy, h, w):
                    car_boxes.append((xyxy, conf))
            elif cls_id == COCO_LIGHT_ID and conf >= LIGHT_MIN_CONF:
                light_boxes.append((xyxy, conf))

        # ── 2. Interior visibility detection ──
        event.interior_visible = self._detect_interior(frame)
        if event.interior_visible:
            self._interior_visible_count += 1
        else:
            self._interior_visible_count = max(0, self._interior_visible_count - 1)

        # ── 3. Distance estimation ──
        nearest_raw_dist = None
        nearest_dist = None
        nearest_box  = None
        for xyxy, conf in car_boxes:
            raw_dist = self._estimate_distance(xyxy)
            if raw_dist is not None and (nearest_raw_dist is None or raw_dist < nearest_raw_dist):
                nearest_raw_dist = raw_dist
                nearest_box  = xyxy

        # Adjust distance based on interior visibility
        if nearest_raw_dist is not None:
            event.raw_nearest_dist_m = nearest_raw_dist
            # If interior is visible (wearable camera), add vehicle front length
            if self._interior_visible_count > 0:
                adjusted_dist = nearest_raw_dist + self.profile.vehicle_front_length_m
                nearest_dist = adjusted_dist
            else:
                # Dashcam mode: use raw distance
                nearest_dist = nearest_raw_dist

        event.nearest_car_dist_m = nearest_dist
        if nearest_dist is not None:
            event.too_close = nearest_dist < self.profile.safe_distance_m

        # ── 4. Traffic-light colour ──
        if light_boxes:
            event.light_detected = True
            best_xyxy = max(light_boxes, key=lambda x: x[1])[0]
            event.light_color = self._classify_light_color(frame, best_xyxy)

        # ── 5. Green-light stationary check ──
        is_moving = self._is_car_moving(frame)
        if event.light_color == "GREEN" and not is_moving and nearest_dist is None:
            self._green_stationary_frames += 1
            if self._green_stationary_frames >= green_frames_threshold:
                event.light_violation = True
        else:
            self._green_stationary_frames = 0

        # ── 6. Lane detection ──
        event.lane_status = self._check_lane(frame)

        # ── 7. Annotate frame ──
        event.annotated_frame = self._annotate(
            frame.copy(), event, car_boxes, light_boxes,
            nearest_box, nearest_dist, h, w
        )

        # Store gray for next motion check
        self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return event

    # ──────────────────────────────────────────
    #  OWN-CAR MASK
    # ──────────────────────────────────────────
    def _is_own_car(self, xyxy, h, w) -> bool:
        """
        Returns True if this detection is likely the user's own bonnet/dashboard.
        Criteria:
          - Bottom of box is in the very bottom strip  AND
          - Horizontal centre is in the centre band (not a car to the side)
        """
        x1, y1, x2, y2 = xyxy
        box_bottom_frac = y2 / h
        cx_frac         = ((x1 + x2) / 2) / w
        return (box_bottom_frac > OWN_CAR_MASK_FRAC_Y and
                OWN_CAR_MASK_FRAC_X[0] < cx_frac < OWN_CAR_MASK_FRAC_X[1])

    # ──────────────────────────────────────────
    #  DISTANCE ESTIMATION  (Triangle Similarity)
    # ──────────────────────────────────────────
    def _estimate_distance(self, xyxy) -> Optional[float]:
        """
        Triangle Similarity Distance Estimation
        ═══════════════════════════════════════
        
        Instead of relying on Y-coordinate (fails on hills/curves),
        we use the bounding box WIDTH and the vehicle's actual width.
        
        Pinhole camera model:
          distance = (focal_length * real_width_m) / bounding_box_width_px
        
        This method is:
          ✓ Robust to road inclines and curves
          ✓ Uses vehicle-specific dimensions from database
          ✓ More accurate for perspective-based distance estimation
        
        Args:
            xyxy: [x1, y1, x2, y2] bounding box coordinates
        
        Returns:
            Estimated distance in metres (rounded to 0.1m precision)
        """
        x1, y1, x2, y2 = xyxy
        bbox_width_px = x2 - x1
        
        # Minimum detectable width (avoid noise)
        if bbox_width_px < 10:
            return None
        
        # Get this vehicle's actual width from database
        vehicle_width = self.profile.vehicle_width_m
        
        # Triangle similarity formula:
        # distance = (f * W) / w
        # where: f = focal constant, W = real width, w = apparent width
        distance_m = (FOCAL_CONST * vehicle_width) / bbox_width_px
        
        return round(distance_m, 1)

    # ──────────────────────────────────────────
    #  TRAFFIC LIGHT COLOUR
    # ──────────────────────────────────────────
    def _classify_light_color(self, frame: np.ndarray, xyxy) -> str:
        """
        Classify traffic light colour by dominant HSV hue in the crop.
        Strategy: split crop into top / middle / bottom thirds and find
        the brightest (most saturated) third, then check its hue.
        """
        x1, y1, x2, y2 = xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "UNKNOWN"

        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_c  = crop.shape[0]

        thirds = [
            hsv[:h_c//3],
            hsv[h_c//3 : 2*h_c//3],
            hsv[2*h_c//3:]
        ]
        names    = ["RED", "YELLOW", "GREEN"]
        # Rough hue ranges: Red ≈ 0-10 or 160-180, Yellow ≈ 20-35, Green ≈ 40-85
        hue_ranges = [
            [(0, 15), (160, 180)],
            [(20, 38)],
            [(38, 90)]
        ]

        scores = []
        for third, ranges in zip(thirds, hue_ranges):
            mask = np.zeros(third.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(third, (lo, 60, 60), (hi, 255, 255))
            scores.append(int(mask.sum()))

        best = int(np.argmax(scores))
        # Require a minimum pixel count to avoid noise
        if scores[best] < 50:
            return "UNKNOWN"
        return names[best]

    # ──────────────────────────────────────────
    #  MOTION DETECTION  (is the user's car moving?)
    # ──────────────────────────────────────────
    def _is_car_moving(self, frame: np.ndarray) -> bool:
        """
        Crude optical-flow proxy: compare centre ROI of current vs previous frame.
        Returns True if significant motion is detected.
        """
        if self._prev_gray is None:
            return True  # assume moving on first frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Use centre strip (avoids edge-of-frame tree/building parallax)
        roi_prev = self._prev_gray[h//4 : 3*h//4, w//4 : 3*w//4]
        roi_curr = gray[h//4 : 3*h//4, w//4 : 3*w//4]
        diff     = cv2.absdiff(roi_prev, roi_curr)
        motion   = float(diff.mean())
        return motion > 1.5   # pixels threshold — tunable

    # ──────────────────────────────────────────
    #  INTERIOR DETECTION  (steering wheel / dashboard)
    # ──────────────────────────────────────────
    def _detect_interior(self, frame: np.ndarray) -> bool:
        """
        Detect if steering wheel or dashboard is visible (wearable camera).
        Strategy:
          1. Check bottom-left and bottom-right corners for dark circular structures (steering wheel)
          2. Check bottom-centre for high-texture areas (dashboard, steering)
          3. Analyse color distribution: interiors have more blacks/dark grays
        Returns True if interior indicators are found.
        """
        h, w = frame.shape[:2]
        
        # Sample three regions: left, centre, right of bottom strip
        bottom_strip_h = int(h * 0.35)   # bottom 35% of frame
        if bottom_strip_h < 10:
            return False
        
        bottom_region = frame[-bottom_strip_h:, :]
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        
        # Detect high-texture areas (steering wheel, dashboard controls)
        # Use Laplacian for edge/texture detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture = np.abs(laplacian).mean()
        
        # Dark pixels indicate interior (steering wheel is typically dark)
        dark_pct = (gray < 80).sum() / gray.size
        
        # Steering wheels are often symmetric circular patterns → check for circular edges
        edges = cv2.Canny(gray, 50, 150)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                    param1=50, param2=30, minRadius=15, maxRadius=80)
        has_circles = circles is not None and len(circles[0]) > 0
        
        # Thresholds for interior detection
        # High texture + dark pixels + circular patterns = likely interior
        interior_score = 0
        if texture > 15:  # significant texture
            interior_score += 1
        if dark_pct > 0.25:  # >25% dark pixels
            interior_score += 1
        if has_circles:  # circular steering wheel detected
            interior_score += 2
        
        return interior_score >= 2

    # ──────────────────────────────────────────
    #  LANE DETECTION
    # ──────────────────────────────────────────
    def _check_lane(self, frame: np.ndarray) -> str:
        """
        Detect lane lines using Canny + Hough.
        Strategy:
          1. Restrict to a trapezoidal ROI (lower half of frame — road surface).
          2. Find left and right line clusters by slope sign.
          3. Compute where the lines intersect the bottom of the frame.
          4. If any line is too close to the horizontal centre → OVER_LINE.
          5. If only one side detected consistently → DRIFT.
        """
        h, w = frame.shape[:2]

        # ROI: trapezoid covering the lower road area
        roi_vertices = np.array([[
            (int(w * 0.0), h),
            (int(w * 0.45), int(h * 0.55)),
            (int(w * 0.55), int(h * 0.55)),
            (int(w * 1.0), h),
        ]], dtype=np.int32)

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur    = cv2.GaussianBlur(gray, (7, 7), 0)
        edges   = cv2.Canny(blur, 40, 120)

        # Mask to ROI
        mask    = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked  = cv2.bitwise_and(edges, mask)

        lines   = cv2.HoughLinesP(
            masked, rho=1, theta=np.pi/180,
            threshold=40, minLineLength=50, maxLineGap=80
        )

        if lines is None:
            return "OK"   # Can't detect → no penalty

        left_xs, right_xs = [], []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Steep enough to be a lane line (|slope| > 0.3)
            if abs(slope) < 0.3:
                continue
            # Extrapolate to bottom of frame
            if abs(slope) > 0.001:
                x_bottom = int(x1 + (h - y1) / slope)
            else:
                x_bottom = x1

            if slope < 0:   # left lane line (negative slope in image coords)
                left_xs.append(x_bottom)
            else:            # right lane line
                right_xs.append(x_bottom)

        left_x  = int(np.mean(left_xs))  if left_xs  else None
        right_x = int(np.mean(right_xs)) if right_xs else None
        centre  = w // 2

        # Over-line: a line is very close to the image centre (car straddling it)
        OVER_THRESH = int(w * 0.08)   # 8% of width
        if left_x  is not None and abs(left_x  - centre) < OVER_THRESH:
            return "OVER_LINE"
        if right_x is not None and abs(right_x - centre) < OVER_THRESH:
            return "OVER_LINE"

        # Drift: only one line visible when both should be (open road)
        if left_xs and not right_xs:
            return "DRIFT"
        if right_xs and not left_xs:
            return "DRIFT"

        return "OK"

    # ──────────────────────────────────────────
    #  ANNOTATION
    # ──────────────────────────────────────────
    def _annotate(self, frame, event: FrameEvent,
                  car_boxes, light_boxes,
                  nearest_box, nearest_dist, h, w) -> np.ndarray:

        # ── draw car boxes ──
        for xyxy, conf in car_boxes:
            dist = self._estimate_distance(xyxy)
            is_nearest = nearest_box is not None and np.array_equal(xyxy, nearest_box)
            color  = (0, 0, 220) if (is_nearest and event.too_close) else (50, 205, 50)
            label  = f"{dist:.0f}m" if dist else "car"
            if is_nearest and event.too_close:
                label += " ⚠ TOO CLOSE"
            cv2.rectangle(frame,
                          (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]),
                          color, 2)
            self._put_label(frame, label, (xyxy[0], xyxy[1] - 8), color)

        # ── draw traffic light boxes ──
        for xyxy, conf in light_boxes:
            color_map = {"RED": (0, 0, 255), "GREEN": (0, 220, 0),
                         "YELLOW": (0, 210, 255), "UNKNOWN": (200, 200, 200)}
            c = color_map.get(event.light_color, (200, 200, 200))
            cv2.rectangle(frame,
                          (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), c, 2)
            self._put_label(frame, f"LIGHT:{event.light_color}",
                            (xyxy[0], xyxy[1] - 8), c)

        # ── HUD overlay (semi-transparent panel) ──
        panel_h = 95
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Distance info
        if nearest_dist is not None:
            safe_d  = self.profile.safe_distance_m
            dist_col = (0, 80, 255) if event.too_close else (60, 220, 60)
            if event.interior_visible and event.raw_nearest_dist_m is not None:
                # Show both raw and adjusted when interior is visible
                dist_txt = f"DIST: {nearest_dist:.0f}m (raw: {event.raw_nearest_dist_m:.0f}m)  |  SAFE: {safe_d:.0f}m  [INTERIOR]"
            else:
                dist_txt = f"DIST: {nearest_dist:.0f}m  |  SAFE: {safe_d:.0f}m"
            cv2.putText(frame, dist_txt, (12, 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, dist_col, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "DIST: --", (12, 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)

        # Lane status
        lane_col = {"OK": (60, 220, 60),
                    "DRIFT": (0, 180, 255),
                    "OVER_LINE": (0, 60, 255)}.get(event.lane_status, (180, 180, 180))
        cv2.putText(frame, f"LANE: {event.lane_status}", (12, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, lane_col, 1, cv2.LINE_AA)

        # Light + violation
        if event.light_detected:
            light_txt = f"LIGHT: {event.light_color}"
            if event.light_violation:
                light_txt += "  !! FAILED TO MOVE !!"
            lc = (0, 0, 255) if event.light_violation else (200, 200, 200)
            cv2.putText(frame, light_txt, (12, 88),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, lc, 1, cv2.LINE_AA)

        # Timestamp
        ts_txt = f"{event.timestamp_s:.1f}s"
        cv2.putText(frame, ts_txt, (w - 100, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)

        return frame

    # ──────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────
    @staticmethod
    def _put_label(frame, text, pos, color, scale=0.5, thick=1):
        """Draw a label with a dark background for readability."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = pos
        cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 2),
                      (10, 10, 10), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)
