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
DriveSense/
├── app.py                  # Gradio UI dashboard & pipeline orchestrator
├── detector.py             # Computer Vision engine (YOLO, lanes, lights, distances, TTC)
├── scorer.py               # Aggregates scores, runs rules, builds reports & timelines
├── user_profile.py         # Manages multiple vehicles & saved historical runs
├── analytics.py            # Generates telemetry plots (Following Distance vs Safe Distance)
├── DriveSense_AI.ipynb     # Jupyter Notebook for running on Google Colab
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

## Where to Run

### ✅ Option A: VS Code / Local Machine (Recommended for local review)

Best for: development, customization, and local evaluation.

1. **Open the project folder** in VS Code.
2. **Open a terminal** in VS Code.
3. **Set up a Virtual Environment** and install dependencies:
   ```bash
   # Create a virtual environment
   python -m venv .venv

   # Activate the virtual environment
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source .venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```
4. **Run the application**:
   In VS Code, open [app.py](file:///c:/SE/DriveSense/app.py) and run it, or execute this in your terminal:
   ```bash
   python app.py
   ```
5. **Accessing the UI**:
   - **Local URL**: Once started, the console will print `Running on local URL: http://localhost:7860`. You can open this in your browser to run locally.
   - **Public URL**: Because `share=True` is enabled in `app.py`, Gradio will also print a **temporary public share link** (e.g., `Running on public URL: https://xxxxxxxxxxxxxx.gradio.live`). Anyone can access this link from any device for up to 72 hours while your local terminal remains running.

*Note: For GPU execution locally, make sure you have PyTorch installed with CUDA support before running `pip install -r requirements.txt`.*

---

### ✅ Option B: Google Colab (Recommended for Faster Processing / No GPU)

Since YOLO model inference and video writing are processor-heavy, running the project on a standard CPU can take up to 10 minutes for longer videos. Google Colab provides a free **T4 GPU** which accelerates processing to near real-time (under a minute).

#### Step-by-Step Google Colab Import & Execution:

1. **Open Google Colab**:
   Go to [colab.research.google.com](https://colab.research.google.com).
2. **Upload the Notebook**:
   - In the popup window, select the **Upload** tab.
   - Choose the [DriveSense_AI.ipynb](file:///c:/SE/DriveSense/DriveSense_AI.ipynb) file from this project folder.
3. **Change Runtime to GPU (CRITICAL for speed)**:
   - In the Colab top menu, go to **Runtime** → **Change runtime type**.
   - In the **Hardware accelerator** dropdown, select **T4 GPU** (do not select CPU as it will be slow).
   - Click **Save**.
4. **Execute the Notebook**:
   - Go to **Runtime** → **Run all** (or press `Ctrl + F9`).
   - The notebook cells will execute sequentially:
     - **Cell 1**: Installs system-level `ffmpeg` (required for output video encoding).
     - **Cell 2**: Installs required python libraries (`ultralytics`, `gradio`, etc.).
     - **Cells 3–7**: Automatically write `detector.py`, `scorer.py`, `user_profile.py`, `analytics.py`, and `app.py` directly into the Colab environment using `%%writefile`.
     - **Cell 8**: Verifies all source files were successfully generated.
     - **Cell 9**: Runs the Gradio app via `!python app.py`.
5. **Open the Dashboard**:
   - Scroll down to the output of the final cell.
   - Look for the line: `Running on public URL: https://xxxxxxxxxxxxxx.gradio.live`.
   - Click this public link. The complete dark-themed DriveSense AI dashboard will open in a new tab, running on Colab's fast GPU back-end!

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
- Speed is not measured directly (requires GPS telemetry); speed limit is given by user.

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
