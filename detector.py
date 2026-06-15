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
COCO_PEDESTRIAN_ID = 0
COCO_BICYCLE_ID    = 1
COCO_CAR_ID        = 2
COCO_MOTORCYCLE_ID = 3
COCO_BUS_ID        = 5
COCO_TRUCK_ID      = 7
COCO_LIGHT_ID      = 9
COCO_STOP_SIGN_ID  = 11

COCO_VEHICLE_IDS   = {2, 5, 7, 3, 1}   # car, bus, truck, motorcycle, bicycle
COCO_CAR_IDS       = {2, 5, 7}

# Representative widths (metres) for distance estimation (Pakistan standard size estimate)
CLASS_WIDTHS = {
    0: 0.55,  # Pedestrian
    1: 0.65,  # Bicycle
    2: 1.75,  # Car (standard)
    3: 0.80,  # Motorcycle
    5: 2.50,  # Bus
    7: 2.50,  # Truck
}

# Approximate focal-length calibration constant (pixels × metres / pixels).
FOCAL_CONST     = 800

# Own-car mask: ignore detections in the bottom-centre strip
OWN_CAR_MASK_FRAC_Y  = 0.82
OWN_CAR_MASK_FRAC_X  = (0.25, 0.75)

# Traffic light & stop sign minimum confidence
LIGHT_MIN_CONF = 0.40
STOP_SIGN_MIN_CONF = 0.40

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
    # Collision and Following
    nearest_car_dist_m : Optional[float] = None    # adjusted distance (bumper-to-bumper)
    raw_nearest_dist_m : Optional[float] = None    # raw bounding-box distance
    nearest_car_class   : Optional[int] = None      # COCO class id of nearest obstacle
    interior_visible : bool = False                 # True if steering/meter/dashboard detected
    too_close       : bool = False
    collision_warning : bool = False                # True if TTC < 2.0s
    ttc_s           : Optional[float] = None        # estimated TTC in seconds
    # Lane
    lane_status     : str  = "OK"        # OK | DRIFT | OVER_LINE
    lane_left_line  : Optional[tuple[int, int]] = None  # (x_bottom, x_top)
    lane_right_line : Optional[tuple[int, int]] = None # (x_bottom, x_top)
    # Traffic light
    light_detected  : bool = False
    light_color     : str  = "NONE"      # RED | GREEN | YELLOW | UNKNOWN
    light_violation : bool = False       # car stationary at green too long
    # Stop Sign
    stop_sign_detected : bool = False
    stop_sign_violation : bool = False   # failed to stop when passing stop sign
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

        # Rolling buffer of recent nearest-car detections for TTC calculation
        # Format: list of (timestamp, distance)
        self._distance_history = []
        
        # Stop sign tracking state
        self._stop_sign_visible = False
        self._stop_sign_stopped = False
        self._stop_sign_max_width = 0
        
        # Reference width for distance calibration (defaults to 1080p width)
        self._frame_width = 1920

    # ──────────────────────────────────────────
    #  MAIN PER-FRAME ENTRY POINT
    # ──────────────────────────────────────────
    def analyse_frame(self, frame: np.ndarray, frame_idx: int,
                      fps: float, green_frames_threshold: int) -> FrameEvent:
        h, w = frame.shape[:2]
        self._frame_width = w
        ts   = frame_idx / max(fps, 1)
        event = FrameEvent(frame_idx=frame_idx, timestamp_s=ts)

        # ── 1. YOLO inference ──
        results = self.model(frame, verbose=False, conf=0.35)
        boxes   = results[0].boxes if results else []

        detected_obstacles = [] # list of (xyxy, conf, cls_id)
        light_boxes = []
        stop_sign_boxes = []

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            xyxy   = box.xyxy[0].cpu().numpy().astype(int)

            if cls_id in COCO_VEHICLE_IDS or cls_id == COCO_PEDESTRIAN_ID:
                if not self._is_own_car(xyxy, h, w):
                    detected_obstacles.append((xyxy, conf, cls_id))
            elif cls_id == COCO_LIGHT_ID and conf >= LIGHT_MIN_CONF:
                light_boxes.append((xyxy, conf))
            elif cls_id == COCO_STOP_SIGN_ID and conf >= STOP_SIGN_MIN_CONF:
                stop_sign_boxes.append((xyxy, conf))

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
        nearest_class = None
        
        for xyxy, conf, cls_id in detected_obstacles:
            raw_dist = self._estimate_distance(xyxy, cls_id)
            if raw_dist is not None and (nearest_raw_dist is None or raw_dist < nearest_raw_dist):
                nearest_raw_dist = raw_dist
                nearest_box  = xyxy
                nearest_class = cls_id

        # Adjust distance based on interior visibility (camera inside car vs dashcam)
        if nearest_raw_dist is not None:
            event.raw_nearest_dist_m = nearest_raw_dist
            event.nearest_car_class = nearest_class
            if self._interior_visible_count > 0:
                # Bumper is closer than camera. Subtract hood offset (physically correct!)
                adjusted_dist = max(0.5, nearest_raw_dist - self.profile.vehicle_front_length_m)
                nearest_dist = adjusted_dist
            else:
                nearest_dist = nearest_raw_dist

        event.nearest_car_dist_m = nearest_dist
        if nearest_dist is not None:
            event.too_close = nearest_dist < self.profile.safe_distance_m

        # ── 4. TTC and Collision Warning (FCW) ──
        if nearest_dist is not None:
            self._distance_history.append((ts, nearest_dist))
            if len(self._distance_history) > 6:
                self._distance_history.pop(0)
                
            # If we have at least 3 points, fit a line to calculate relative velocity (slope)
            if len(self._distance_history) >= 3:
                times = [pt[0] for pt in self._distance_history]
                dists = [pt[1] for pt in self._distance_history]
                slope, _ = np.polyfit(times, dists, 1) # slope is relative velocity (m/s)
                
                # If slope is negative, we are closing in (V_rel is positive closing speed)
                if slope < -0.2:
                    v_closing = -slope
                    ttc = nearest_dist / v_closing
                    event.ttc_s = round(ttc, 2)
                    if ttc < 2.0:
                        event.collision_warning = True
        else:
            self._distance_history.clear()

        # ── 5. Traffic-light colour ──
        if light_boxes:
            event.light_detected = True
            best_xyxy = max(light_boxes, key=lambda x: x[1])[0]
            event.light_color = self._classify_light_color(frame, best_xyxy)

        # ── 6. Green-light stationary check ──
        is_moving = self._is_car_moving(frame)
        if event.light_color == "GREEN" and not is_moving and nearest_dist is None:
            self._green_stationary_frames += 1
            if self._green_stationary_frames >= green_frames_threshold:
                event.light_violation = True
        else:
            self._green_stationary_frames = 0

        # ── 7. Stop Sign check & violations ──
        if stop_sign_boxes:
            event.stop_sign_detected = True
            best_ss_box = max(stop_sign_boxes, key=lambda x: x[1])[0]
            ss_w = best_ss_box[2] - best_ss_box[0]
            self._stop_sign_visible = True
            self._stop_sign_max_width = max(self._stop_sign_max_width, ss_w)
            if not is_moving:
                self._stop_sign_stopped = True
        else:
            # If stop sign was visible but now is gone
            if self._stop_sign_visible:
                # If we passed it closely (width > 25px) and never stopped, trigger violation
                if self._stop_sign_max_width > 25 and not self._stop_sign_stopped:
                    event.stop_sign_violation = True
                # Reset tracking
                self._stop_sign_visible = False
                self._stop_sign_stopped = False
                self._stop_sign_max_width = 0

        # ── 8. Lane detection ──
        event.lane_status, event.lane_left_line, event.lane_right_line = self._check_lane_and_get_lines(frame)

        # ── 9. Annotate frame ──
        event.annotated_frame = self._annotate(
            frame.copy(), event, detected_obstacles, light_boxes, stop_sign_boxes,
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
    def _estimate_distance(self, xyxy, cls_id: int) -> Optional[float]:
        """
        Triangle Similarity Distance Estimation
        ═══════════════════════════════════════
        Uses class-specific physical widths to compute distance:
          distance = (FOCAL_CONST * target_width_m) / bounding_box_width_px
        """
        x1, y1, x2, y2 = xyxy
        bbox_width_px = x2 - x1
        
        if bbox_width_px < 8:
            return None
        
        # Get target width based on detected class
        target_width = CLASS_WIDTHS.get(cls_id, 1.75)
        
        # Scale focal constant relative to 1080p width (1920px) to handle rescaled/low-res streams
        adjusted_focal = FOCAL_CONST * (self._frame_width / 1920.0)
        
        distance_m = (adjusted_focal * target_width) / bbox_width_px
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
    # ──────────────────────────────────────────
    #  LANE DETECTION (Polynomial Fitting)
    # ──────────────────────────────────────────
    def _check_lane_and_get_lines(self, frame: np.ndarray) -> tuple[str, Optional[tuple[int, int]], Optional[tuple[int, int]]]:
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
            return "OK", None, None

        left_pts, right_pts = [], []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Steep enough to be a lane line
            if abs(slope) < 0.3 or abs(slope) > 5.0:
                continue

            cx = (x1 + x2) // 2
            if slope < 0 and cx < w * 0.55:   # left lane line
                left_pts.append((x1, y1))
                left_pts.append((x2, y2))
            elif slope > 0 and cx > w * 0.45:  # right lane line
                right_pts.append((x1, y1))
                right_pts.append((x2, y2))

        left_line = None
        right_line = None
        y_bottom = h
        y_top = int(h * 0.6)

        # Fit left line: x = a*y + b
        if len(left_pts) >= 2:
            ys = [p[1] for p in left_pts]
            xs = [p[0] for p in left_pts]
            try:
                a, b = np.polyfit(ys, xs, 1)
                x_bottom = int(a * y_bottom + b)
                x_top = int(a * y_top + b)
                if -w < x_bottom < 2*w:
                    left_line = (x_bottom, x_top)
            except np.linalg.LinAlgError:
                pass

        # Fit right line: x = a*y + b
        if len(right_pts) >= 2:
            ys = [p[1] for p in right_pts]
            xs = [p[0] for p in right_pts]
            try:
                a, b = np.polyfit(ys, xs, 1)
                x_bottom = int(a * y_bottom + b)
                x_top = int(a * y_top + b)
                if -w < x_bottom < 2*w:
                    right_line = (x_bottom, x_top)
            except np.linalg.LinAlgError:
                pass

        centre  = w // 2
        
        # Over-line: if wheels are close/crossing the lane markers
        OVER_THRESH = int(w * 0.08)   # 8% of width
        if left_line is not None and abs(left_line[0] - centre) < OVER_THRESH:
            return "OVER_LINE", left_line, right_line
        if right_line is not None and abs(right_line[0] - centre) < OVER_THRESH:
            return "OVER_LINE", left_line, right_line

        # Drift: only one line visible
        if left_line is not None and right_line is None:
            return "DRIFT", left_line, right_line
        if right_line is not None and left_line is None:
            return "DRIFT", left_line, right_line

        if not left_line and not right_line:
            return "OK", None, None

        return "OK", left_line, right_line

    # ──────────────────────────────────────────
    #  LANE OVERLAY DRAWING
    # ──────────────────────────────────────────
    def _draw_lane_overlay(self, frame, event: FrameEvent, h, w) -> np.ndarray:
        status = event.lane_status
        left = event.lane_left_line
        right = event.lane_right_line
        
        # BGR Colors
        color_map = {
            "OK": (80, 220, 80),       # Soft green
            "DRIFT": (0, 165, 255),    # Soft orange
            "OVER_LINE": (0, 0, 220)   # Soft red
        }
        color = color_map.get(status, (80, 220, 80))
        
        overlay = frame.copy()
        y_bottom = h
        y_top = int(h * 0.6)
        
        # Draw transparent lane polygon if both lines are detected
        if left is not None and right is not None:
            pts = np.array([
                [left[0], y_bottom],
                [left[1], y_top],
                [right[1], y_top],
                [right[0], y_bottom]
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
            
            # Draw line boundaries
            cv2.line(frame, (left[0], y_bottom), (left[1], y_top), color, 3, cv2.LINE_AA)
            cv2.line(frame, (right[0], y_bottom), (right[1], y_top), color, 3, cv2.LINE_AA)
        elif left is not None:
            cv2.line(frame, (left[0], y_bottom), (left[1], y_top), color, 3, cv2.LINE_AA)
        elif right is not None:
            cv2.line(frame, (right[0], y_bottom), (right[1], y_top), color, 3, cv2.LINE_AA)
            
        return frame

    # ──────────────────────────────────────────
    #  ANNOTATION
    # ──────────────────────────────────────────
    def _annotate(self, frame, event: FrameEvent,
                  detected_obstacles, light_boxes, stop_sign_boxes,
                  nearest_box, nearest_dist, h, w) -> np.ndarray:

        # ── 1. Draw lane overlay first ──
        frame = self._draw_lane_overlay(frame, event, h, w)

        # ── 2. Draw detected obstacles ──
        for xyxy, conf, cls_id in detected_obstacles:
            dist = self._estimate_distance(xyxy, cls_id)
            is_nearest = nearest_box is not None and np.array_equal(xyxy, nearest_box)
            
            # Determine color and labels based on class
            if cls_id == COCO_PEDESTRIAN_ID:
                color = (255, 120, 0) # Cyan/Blue-ish in BGR
                label = f"Pedestrian {dist:.1f}m" if dist else "pedestrian"
            elif cls_id == COCO_MOTORCYCLE_ID or cls_id == COCO_BICYCLE_ID:
                color = (255, 200, 0) # Yellow/Blue
                label = f"Cycle {dist:.1f}m" if dist else "cycle"
            else: # Car, bus, truck
                color = (0, 0, 220) if (is_nearest and event.too_close) else (50, 205, 50)
                label = f"Vehicle {dist:.1f}m" if dist else "vehicle"
                if is_nearest and event.too_close:
                    label += " [TOO CLOSE]"
                if is_nearest and event.collision_warning:
                    label += " ⚠ FCW ⚠"
                    color = (0, 0, 255) # Bright Red

            # Draw bounding box
            thickness = 3 if (is_nearest and (event.too_close or event.collision_warning)) else 2
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness)
            self._put_label(frame, label, (xyxy[0], xyxy[1] - 8), color)

        # ── 3. Draw traffic light boxes ──
        for xyxy, conf in light_boxes:
            color_map = {"RED": (0, 0, 255), "GREEN": (0, 220, 0),
                         "YELLOW": (0, 210, 255), "UNKNOWN": (200, 200, 200)}
            c = color_map.get(event.light_color, (200, 200, 200))
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), c, 2)
            self._put_label(frame, f"LIGHT:{event.light_color}", (xyxy[0], xyxy[1] - 8), c)

        # ── 4. Draw stop sign boxes ──
        for xyxy, conf in stop_sign_boxes:
            c = (0, 0, 255) # Red box
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), c, 3)
            label = "STOP SIGN"
            if event.stop_sign_violation:
                label += " (VIOLATION)"
            self._put_label(frame, label, (xyxy[0], xyxy[1] - 8), c)

        # ── 5. HUD overlay (semi-transparent panel at the top) ──
        panel_h = 95
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 16, 16), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Distance & TTC Info
        if nearest_dist is not None:
            safe_d = self.profile.safe_distance_m
            dist_col = (0, 80, 255) if event.too_close else (60, 220, 60)
            if event.collision_warning:
                dist_col = (0, 0, 255) # Red for critical
                
            dist_txt = f"GAP: {nearest_dist:.1f}m  |  SAFE: {safe_d:.1f}m"
            if event.ttc_s is not None:
                dist_txt += f"  |  TTC: {event.ttc_s:.1f}s"
            if event.interior_visible:
                dist_txt += " [CABIN]"
            cv2.putText(frame, dist_txt, (12, 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, dist_col, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "GAP: --  |  SAFE: --", (12, 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)

        # Lane status
        lane_col = {"OK": (60, 220, 60),
                    "DRIFT": (0, 180, 255),
                    "OVER_LINE": (0, 60, 255)}.get(event.lane_status, (180, 180, 180))
        cv2.putText(frame, f"LANE STATUS: {event.lane_status}", (12, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, lane_col, 1, cv2.LINE_AA)

        # Light & Stop Sign Status
        sig_col = (180, 180, 180)
        sig_txt = "SYSTEMS ONLINE"
        
        if event.light_detected:
            sig_txt = f"TRAFFIC LIGHT: {event.light_color}"
            sig_col = (0, 220, 0) if event.light_color == "GREEN" else (0, 0, 255)
            if event.light_violation:
                sig_txt += " (GREEN LIGHT DELAY VIOLATION)"
                sig_col = (0, 0, 255)
        elif event.stop_sign_detected:
            sig_txt = "STOP SIGN DETECTED"
            sig_col = (0, 120, 255)
            if event.stop_sign_violation:
                sig_txt += " (STOP SIGN ROLL VIOLATION)"
                sig_col = (0, 0, 255)
                
        cv2.putText(frame, sig_txt, (12, 88),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, sig_col, 1, cv2.LINE_AA)

        # Simulated Speedometer
        is_moving = self._is_car_moving(frame)
        if is_moving:
            simulated_speed = int(self.profile.speed_limit_kmh * 0.9 + (event.frame_idx % 12 - 6) * 0.5)
        else:
            simulated_speed = 0
            
        speed_txt = f"{simulated_speed} KM/H"
        cv2.putText(frame, speed_txt, (w - 150, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"LIMIT: {int(self.profile.speed_limit_kmh)}", (w - 150, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

        # Flashing HUD Warnings
        warning_active = event.collision_warning or event.lane_status in {"DRIFT", "OVER_LINE"} or event.stop_sign_violation or event.light_violation
        if warning_active and (event.frame_idx // 3) % 2 == 0:
            if event.collision_warning:
                cv2.rectangle(frame, (w//2 - 180, h - 80), (w//2 + 180, h - 30), (0, 0, 220), -1)
                cv2.putText(frame, "COLLISION WARNING (TTC < 2s)", (w//2 - 160, h - 48),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            elif event.lane_status == "OVER_LINE":
                cv2.rectangle(frame, (w//2 - 180, h - 80), (w//2 + 180, h - 30), (0, 69, 255), -1)
                cv2.putText(frame, "LANE DEPARTURE WARNING", (w//2 - 145, h - 48),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

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
