# ⬡ DriveSense AI — Intelligent Dashcam Analysis System
**ITU Lahore · BSAI · Software Engineering Final Project**  
Team: Qasim Bin Shahzad (BSAI-24070) · Salman Ejaz Jathol (BSAI-24067)

---

## Project Overview

DriveSense AI is a computer-vision-based driving analysis tool that takes dashcam or wearable-camera footage and produces:

- An annotated output video with real-time overlays
- A numeric safety score (0.0 – 10.0) broken down into three categories
- A personalised improvement report

---

## Features

| Feature | How it works |
|---|---|
| **Vehicle / car detection** | YOLOv11 (COCO classes: car, bus, truck). Own-car mask filters bonnet/dashboard detections in the bottom-centre strip. |
| **Real-world distance estimation** | Pinhole camera model: `d = focal_const / apparent_width_px`. Calibrated for typical 1080p dashcams. |
| **Safe distance calculation** | Physics-based: reaction distance + braking distance, scaled by the user's braking profile and a 1.2× leniency factor per Pakistani traffic norms. |
| **Lane detection** | Canny edge → HoughLinesP in a trapezoidal road ROI. Detects DRIFT (one line only visible) and OVER_LINE (tyre on the cat's-eye / marking). |
| **Traffic light detection** | YOLOv11 class 9. Colour classified via HSV hue analysis of the bulb region (top=Red, middle=Yellow, bottom=Green). |
| **Green-light violation** | If the user's car is stationary for ≥ 2 seconds after a green light is visible (and no car is blocking), a violation is logged. |
| **Custom braking profile** | User enters 0–100 km/h braking time; safe distance adjusts automatically. |
| **Safety scoring** | Weighted: Distance 40%, Lane 35%, Lights 25%. |

---

## File Structure

```
drivesense/
├── app.py           # Gradio UI + pipeline orchestration
├── detector.py      # All CV & YOLO logic (detection, distance, lanes, lights)
├── scorer.py        # Score aggregation, report generation
├── requirements.txt
└── README.md
```

---

## Where to Run

### ✅ Option A: VS Code / Local Machine (Recommended for submission)

Best for: final demo, stable URLs, reproducible results.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py
```

Open `http://localhost:7860` in your browser.  
A public Gradio share URL is also printed — paste this as your submission URL.

**GPU (recommended):** If you have an NVIDIA GPU, install the CUDA-compatible torch before the others:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python app.py
```

---

### ✅ Option B: Google Colab

Use this if you don't have a GPU locally. Colab's T4 GPU is free and significantly faster.

```python
# Cell 1 — install
!pip install ultralytics gradio opencv-python-headless numpy

# Cell 2 — upload files
from google.colab import files
files.upload()   # upload app.py, detector.py, scorer.py

# Cell 3 — run
!python app.py
```

Colab will print a `gradio.live` public URL automatically (because `share=True` is set).

**Note:** Colab sessions expire after ~12 hours. For a stable submission URL use VS Code + a persistent machine or Hugging Face Spaces.

---

## Deploying to Hugging Face Spaces (Permanent Submission Link)

Hugging Face Spaces provides a free, permanent public URL for your Gradio app (e.g., `https://huggingface.co/spaces/YOUR_USERNAME/drivesense`). This is ideal for final project submission.

1. **Create Account**: Register a free account at [huggingface.co](https://huggingface.co).
2. **Create Space**: Click **New Space** → name it `drivesense` → select **Gradio** as the SDK.
3. **Upload Files**: Upload the following files to your space repository:
   - `app.py`
   - `detector.py`
   - `scorer.py`
   - `user_profile.py`
   - `analytics.py`
   - `requirements.txt`
4. **App Build**: Hugging Face will automatically download dependencies and start the app.
5. **GPU Acceleration (Optional)**: For near real-time processing, go to the Space **Settings** and select the **T4 Small GPU** (free/affordable tier) to accelerate YOLO inferences.

---

## Safety Score Formula

```
overall = 0.40 × distance_score
        + 0.35 × lane_score
        + 0.25 × light_score

distance_score = 10 – min(10, (too_close_frames / car_visible_frames) × 18)
lane_score     = 10 – (drifts × 0.8) – (over_lines × 1.2)
light_score    = 10 – (violations × 2.5)
```

Safe distance formula (from DriverProfile):
```
v = speed_limit / 3.6               # m/s
a = (100/3.6) / braking_time_s     # deceleration m/s²
d = (v × 1.5 + v² / (2a)) × 1.2   # metres
```

---

## Known Limitations

- Distance estimation uses a fixed focal constant calibrated for ~1080p 60° HFOV dashcams. For fisheye or wide-angle lenses, tune `FOCAL_CONST` in `detector.py`.
- Lane detection works best on clearly marked roads in daylight. Poor marking, rain, or night conditions reduce accuracy.
- Traffic light colour classification can fail if the light is very small in frame or heavily overexposed.
- Speed is not measured directly (requires GPS telemetry); speed limit is user-supplied.

---

## SRS Traceability

| SRS Requirement | Implementation |
|---|---|
| Upload MP4/AVI footage | `gr.Video` input in `app.py` |
| Safety score 0.0–10.0 | `scorer.py` → `SafetyReport.overall_score` |
| Detect brake lights / reaction time | Distance drop detection between frames (nearest_car_dist_m delta) |
| Lane boundary monitoring | `detector._check_lane()` |
| Speedometer/RPM tips | Speed limit entered by user; braking profile used for personalised tips |
| Min 15 FPS processing | `DESIRED_FPS=12` on CPU; GPU runs faster |
| Signal detection in daylight | HSV colour classification in `_classify_light_color()` |
| Use case: Driver Reaction Analysis | `light_violation` + `too_close` flags → scorer penalties |

---

*DriveSense AI — for educational use only.*
