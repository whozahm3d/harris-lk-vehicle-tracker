# Harris–LK Vehicle Tracker

> Classical single-object vehicle tracking using Harris Corner Detection and pyramidal Lucas–Kanade Optical Flow — no deep learning, no Kalman filter.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078D6?style=flat-square&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)]()
[![Course](https://img.shields.io/badge/Course-Fundamentals%20of%20Computer%20Vision-orange?style=flat-square)]()

---

## Overview

This project implements a complete, interpretable single-object tracker built entirely on classical computer vision primitives. Given a user-defined bounding box on a start frame, the system detects Harris corners inside the box and tracks them across subsequent frames using pyramidal LK optical flow — updating the bounding box each frame via the median displacement of surviving feature points.

The design deliberately avoids deep learning and Kalman filtering to demonstrate that well-engineered classical methods can achieve robust, stable tracking on high-resolution 4K footage when the right fallback mechanisms are in place.

**Validated on two 4K test sequences — 100% tracker stability on both, across 280 and 529 frames respectively.**

---

## Table of Contents

- [Demo](#demo)
- [How It Works](#how-it-works)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration Reference](#configuration-reference)
- [Results](#results)
- [Tested Environment](#tested-environment)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Academic Context](#academic-context)
- [Citation](#citation)
- [License](#license)

---

## Demo

### Test 1 — Top-Down Drone Footage (4K, 24 fps, 280 frames)

Tracking a silver car on a road from directly overhead. The uniform car roof produces sparse feature points (~66 at initialisation) that compete with strong stationary road-marking corners. The velocity-guided background filter suppresses the road noise and keeps tracking stable across the full sequence.

![Test 1 Tracking Snapshots](results/test_1_20260516_173936/plots/integration_snapshots.png)

---

### Test 2 — Oblique Drone Surveillance (4K, 30 fps, 529 frames)

Tracking a vehicle from an oblique angle across a rural road. The perspective exposes the car's side panels, windows, and wheel arches, yielding 137 stable corner points throughout all 529 frames.

![Test 2 Tracking Snapshots](results/test_2_20260516_170555/plots/integration_snapshots.png)

---

### Harris Corner Detection at Initialisation

66 Harris corners detected inside the manually drawn bounding box on frame 0. Points cluster on the car's edges, roof panel boundaries, and windows — exactly where structural gradient information is richest.

![Harris Corners Detected](results/test_1_20260516_173936/plots/phase_2/harris_corners.png)

---

### LK Optical Flow — Frame-to-Frame Tracking

All 66 corners successfully tracked on the first inter-frame pass. Yellow points show tracked feature locations; the bounding box (green) is updated via median displacement of all surviving points.

![LK Optical Flow](results/test_1_20260516_173936/plots/phase_3/lk_tracking.png)

---

### Harris Corner Survival — Test 1 (280 frames)

Point count across the full sequence. The tracker never drops below the 4-point minimum threshold (red dashed line). Stepped drops correspond to forward-backward filtering events; redetection fires at the 25% survival threshold when needed.

![Point Survival](results/test_1_20260516_173936/plots/phase_4/point_survival.png)

---

### Object Centroid Trajectory — Test 1

Full centroid path across 280 frames, colour-mapped by frame index (purple → yellow). The trajectory accurately traces the car's route: straight road section, right-angle turn at intersection, and continued travel.

![Centroid Trajectory](results/test_1_20260516_173936/plots/phase_4/centroid_trajectory.png)

> **Note:** The `results/` directory is excluded from this repository (see `.gitignore`). The images above are generated automatically when you run the pipeline. Paths will be valid after your first run.

---

## How It Works

The tracker runs a strict 6-step pipeline per frame:

| Step | Operation | Detail |
|------|-----------|--------|
| 1 | **Read frame** | Decode next frame from video |
| 2 | **Detect Harris corners** | Run inside bounding box on start frame only (or on redetection trigger) |
| 3 | **Forward LK pass** | Track all points from previous frame to current frame |
| 4 | **Backward LK pass** | Track points back from current to previous frame |
| 5 | **Forward-backward filter** | Reject any point whose round-trip error exceeds 3.5 px |
| 6 | **Update bounding box** | Shift bbox by the median displacement of surviving points |

This loop repeats for every frame until the sequence ends, the user quits, or the tracker deactivates after 45 consecutive zero-point frames.

---

## Key Engineering Decisions

These are the mechanisms that make the difference between a tracker that works on clean test cases and one that holds up on real 4K footage.

### 1. Velocity-Guided Background Filter

**Problem:** When tracking a car moving at speed, many Harris corners belong to stationary road markings and pavement — not the vehicle. These points produce near-zero optical flow, which contaminates the median displacement estimate and causes bbox drift.

**Solution:** When the estimated object speed exceeds 3 px/frame, any point whose displacement magnitude falls below 30% of the current velocity estimate is classified as background and discarded before the median is computed.

**Result:** Background corner contamination is eliminated without tuning per-video thresholds.

### 2. Survival-Based Redetection

**Problem:** Scheduled redetection (every N frames) wastes computation when points are healthy and fires too late when points degrade suddenly.

**Solution:** Redetection triggers only when surviving point count falls below 25% of the initial count. Harris corners are re-detected inside the current bounding box estimate, replenishing the feature pool on demand.

**Result:** The tracker adapts to texture changes and partial occlusion without a fixed schedule.

### 3. Velocity Prediction Fallback

**Problem:** When a vehicle exits the frame or passes through a brief occlusion, point count can drop to zero — causing the bbox to freeze at its last known position.

**Solution:** An exponentially smoothed velocity estimate is maintained every frame. When active point count drops critically low, bbox position is updated by blending the velocity estimate with the measured displacement. When zero points survive, pure velocity extrapolation is used for up to 45 frames.

**Result:** The tracker follows the vehicle to the frame boundary rather than stalling at the last observed position.

### 4. CLAHE Preprocessing

**Problem:** 4K drone footage often has uneven illumination across the frame, which degrades both Harris detection (misses corners in dark regions) and LK convergence (poor gradient estimates in overexposed areas).

**Solution:** Contrast Limited Adaptive Histogram Equalisation (CLAHE) is applied to the grayscale frame before both Harris detection and LK tracking. Clip limit 3.5, tile grid 4×4.

**Result:** Corner detection and flow estimation are more uniform across lighting conditions without amplifying noise globally.

---

## Project Structure

```
harris-lk-vehicle-tracker/
│
├── core/
│   ├── harris_detector.py        # Harris corner detection with CLAHE preprocessing and bbox mask
│   ├── lucas_kanade.py           # Pyramidal LK with forward-backward error filtering
│   └── object_tracker.py         # Stateful tracker: velocity prediction, redetection, bbox update
│
├── utils/
│   ├── config_loader.py          # YAML config loader with dot-access attribute interface
│   ├── folder_manager.py         # Timestamped output directory manager
│   ├── health_check.py           # Pre-flight environment and dependency validation
│   └── resolution_utils.py       # Frame resolution helpers
│
├── metrics/
│   ├── logger.py                 # Levelled logging (INFO / DEBUG / WARNING)
│   └── performance_tracker.py    # Per-frame metric accumulation and JSON export
│
├── visualization/
│   ├── display.py                # Bounding box, centroid trail, and corner overlay rendering
│   └── plotter.py                # Result plot generation (Harris response, trajectory, survival)
│
├── video_io/
│   ├── output_writer.py          # OpenCV VideoWriter wrapper (H.264 / MP4)
│   └── video_source.py           # Video source with buffering
│
├── notebooks/
│   ├── phase1_infrastructure.ipynb   # Config loading, logging, folder management
│   ├── phase2_algorithms.ipynb       # Harris detection + interactive bbox selection via cv2.selectROI
│   ├── phase3_tracking.ipynb         # LK optical flow + forward-backward filtering
│   ├── phase4_results.ipynb          # Evaluation metrics, plots, analysis
│   └── phase5_integration.ipynb      # Full end-to-end pipeline (primary entry point)
│
├── results/                      # Auto-generated — excluded from repo (.gitignore)
├── videos/                       # Place test videos here — excluded from repo (.gitignore)
├── report/
│   └── cv_project_tracking_report.pdf
│
├── main.py                       # CLI entry point
├── config.yaml                   # All tunable parameters
├── requirements.txt
└── LICENSE
```

---

## Installation

### Prerequisites

- Python 3.12
- Windows 11 (primary supported platform — see [Troubleshooting](#troubleshooting) for Linux/macOS notes)
- Jupyter Notebook or JupyterLab

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/whozahm3d/harris-lk-vehicle-tracker.git
cd harris-lk-vehicle-tracker
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the OpenH264 DLL (Windows only)**

Required for H.264 MP4 output via OpenCV on Windows. Download `openh264-1.8.0-win64.dll` from the [Cisco OpenH264 releases page](https://github.com/cisco/openh264/releases) and place it in the **project root** alongside `main.py`.

**5. Create the videos folder and add your test video**

```bash
mkdir videos
# Place your video file inside: videos/your_video.mp4
```

---

## Usage

### Step 1 — Configure `config.yaml`

Set the three input fields for your video. Everything else uses the defaults from the configuration reference below.

```yaml
input:
  source: "C:\\path\\to\\your\\video\\test_1.mp4"   # Full path to video file
  start_frame: 0                                      # Frame index to begin tracking from
  bbox: [x, y, width, height]                         # Bounding box in full-resolution pixels
```

> **Windows path note:** Use double backslashes (`\\`) or forward slashes (`/`) — single backslashes will fail silently.

### Step 2 — Select your start frame and draw the bounding box

Open and run `notebooks/phase2_algorithms.ipynb`. It will:

- Open a frame scrubber window to navigate the video and select the exact start frame
- Launch `cv2.selectROI` to draw a bounding box around the target vehicle
- Print the exact `start_frame` and `bbox` values to copy into `config.yaml`

### Step 3 — Run the full tracking pipeline

Open and run `notebooks/phase5_integration.ipynb`. This will:

- Load config, initialise the tracker on the selected frame, and process all subsequent frames
- Display a live 1280×720 preview window during tracking (press `Q` to stop early)
- Save the annotated output video to `results/<video_name>_<timestamp>/videos/`
- Save per-run logs, metrics JSON, parameter snapshot, and diagnostic plots

### CLI Alternative

If you prefer not to use Jupyter:

```bash
python main.py --config config.yaml
```

### Output Structure

Each run produces an isolated, timestamped folder — successive runs never overwrite previous results.

```
results/
└── test_1_20260516_173936/
    ├── videos/
    │   └── test_1_tracked.mp4          # Annotated output video (H.264)
    ├── logs/
    │   └── run.log                     # Full run log with per-frame point counts
    ├── plots/
    │   ├── phase_2/
    │   │   ├── first_frame.png
    │   │   ├── harris_corners.png
    │   │   ├── harris_response.png
    │   │   └── quality_level_comparison.png
    │   ├── phase_3/
    │   │   ├── lk_tracking.png
    │   │   ├── fb_error_distribution.png
    │   │   └── bbox_update.png
    │   ├── phase_4/
    │   │   ├── centroid_trajectory.png
    │   │   ├── point_survival.png
    │   │   └── snapshots.png
    │   └── integration_snapshots.png
    ├── metrics/
    │   └── test_1_metrics.json         # Avg / min / max points, total frame count
    └── params/
        └── test_1_params.json          # Full config snapshot for this run
```

---

## Configuration Reference

<details>
<summary>Click to expand</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Input** | | |
| `input.source` | `""` | Full path to the input video file |
| `input.start_frame` | `0` | Frame index to begin tracking from |
| `input.bbox` | `[x, y, w, h]` | Bounding box in full-resolution pixels — set via Phase 2 notebook |
| **Harris Corner Detection** | | |
| `harris.max_corners` | `400` | Maximum corners to detect inside the bounding box |
| `harris.quality_level` | `0.005` | Minimum corner quality relative to the strongest corner — lower yields more points |
| `harris.min_distance` | `3` | Minimum pixel distance between detected corners |
| `harris.block_size` | `5` | Neighbourhood size for computing the corner response matrix |
| `harris.k` | `0.04` | Harris sensitivity constant — standard value, rarely needs tuning |
| `harris.redetect_threshold` | `0.25` | Survival ratio below which redetection fires (0.25 = below 25% of initial count) |
| **Lucas–Kanade Optical Flow** | | |
| `lucas_kanade.win_size` | `[41, 41]` | Search window size — larger values handle faster inter-frame motion |
| `lucas_kanade.max_level` | `5` | Pyramid levels — higher values handle larger displacements |
| `lucas_kanade.max_iter` | `30` | Maximum LK iterations per point per pyramid level |
| `lucas_kanade.epsilon` | `0.01` | LK convergence criterion |
| `lucas_kanade.min_eig_threshold` | `0.001` | Minimum eigenvalue threshold — discards poorly conditioned texture patches |
| `lucas_kanade.fb_error_threshold` | `3.5` | Forward-backward round-trip error threshold in pixels — points above this are rejected |
| **Tracking Logic** | | |
| `tracking.min_points_per_object` | `4` | Minimum surviving points required for a valid median displacement estimate |
| `tracking.bbox_smoothing` | `0.0` | Temporal smoothing on bbox updates — 0.0 = snap to median exactly |
| `tracking.redetect_interval` | `60` | Scheduled redetection interval (reserved — currently unused; survival-based only) |
| **Memory** | | |
| `memory.max_trail_length` | `60` | Maximum centroid positions stored in the trail deque |
| `memory.max_inactive_frames` | `45` | Zero-point frames before tracker deactivates (~1.5 s at 30 fps) |
| **Preprocessing** | | |
| `preprocessing.clahe_enable` | `true` | Enable CLAHE contrast normalisation before Harris detection and LK tracking |
| `preprocessing.clahe_clip_limit` | `3.5` | CLAHE clip limit — higher values increase local contrast enhancement |
| `preprocessing.clahe_tile_size` | `[4, 4]` | CLAHE tile grid size for local histogram equalisation |
| **Output** | | |
| `output.save_video` | `true` | Save annotated output video to the results folder |
| `output.show_fps` | `true` | Render FPS overlay on the output video |
| `output.show_trails` | `true` | Render centroid trail on the output video |
| **Metrics** | | |
| `metrics.log_interval` | `30` | Log per-frame point count every N frames |

</details>

---

## Results

Evaluated on two 4K test sequences. Both runs completed with full tracker stability — the bbox never lost the target vehicle and never required manual intervention.

| Metric | Test 1 — Drone (Top-Down) | Test 2 — Drone (Oblique) |
|--------|--------------------------|--------------------------|
| Resolution | 3840 × 2160 | 3840 × 2160 |
| Frame Rate | 24 fps | 30 fps |
| Total Frames | 280 | 529 |
| Initial Corners | 66 | 137 |
| Avg Points / Frame | 40.89 | 137.0 |
| Min Points / Frame | 36 | 137 |
| Max Points / Frame | 64 | 137 |
| Tracker Stability | ✅ 100% | ✅ 100% |
| Processing Rate | ~1.63 fps (CPU only) | ~1.25 fps (CPU only) |
| Primary Challenge | Background corner competition | Frame-exit degradation |
| Key Mechanism | Velocity-guided background filter | Velocity prediction + exit threshold |

**Test 1** — Top-down drone view of a silver car. The uniform roof surface produces limited texture diversity, resulting in ~66 initial corners. Stationary road-marking corners compete with the vehicle throughout the sequence. The velocity-guided filter suppresses them successfully across all 280 frames.

**Test 2** — Oblique drone view of a vehicle on a rural road over 529 frames. The perspective angle reveals the car's side panels, windows, and wheel arches — yielding 137 stable corners that remain constant throughout, indicating no redetection events were needed at any point.

> All results obtained on CPU only. No GPU acceleration was used.

---

## Tested Environment

| Component | Version |
|-----------|---------|
| OS | Windows 11 |
| Python | 3.12 |
| OpenCV | 4.9.0.80 |
| NumPy | 1.26.4 |
| PyYAML | 6.0.1 |
| Matplotlib | 3.8.3 |
| Pandas | 2.2.1 |
| SciPy | 1.12.0 |
| tqdm | 4.66.2 |
| GPU | Not required |

---

## Troubleshooting

**`Failed to initialize VideoWriter` or blank/corrupt output video**

The OpenH264 DLL is missing from the project root.

```
Solution: Download openh264-1.8.0-win64.dll from:
https://github.com/cisco/openh264/releases
Place it in the project root directory alongside main.py.
```

---

**`Cannot open video: path/to/video.mp4`**

The video path in `config.yaml` is wrong or the file does not exist at that path.

```yaml
# Correct — double backslashes on Windows
source: "C:\\Users\\yourname\\harris-lk-vehicle-tracker\\videos\\test_1.mp4"

# Also correct — forward slashes work on Windows too
source: "C:/Users/yourname/harris-lk-vehicle-tracker/videos/test_1.mp4"

# Wrong — single backslashes will fail
source: "C:\Users\yourname\videos\test_1.mp4"
```

---

**`AttributeError: module 'ctypes' has no attribute 'windll'`**

You are running on Linux or macOS. The DPI awareness call in `main.py` is Windows-only. It is already wrapped in a try/except block — if you see this error, verify your Python version is 3.12 and re-run. No code changes should be required.

---

**Tracker drifts off the vehicle after a sharp turn**

The bounding box is axis-aligned and fixed in size. A vehicle turning sharply will have its actual silhouette misaligned with the bbox. Reduce `harris.quality_level` (e.g. to `0.003`) to detect more corners on the vehicle body, which improves displacement estimation during turns.

---

**Processing is too slow for real-time use**

Expected. The system processes 4K footage at ~1.3–1.6 fps on CPU. To increase speed: (1) downscale the video before running, or (2) reduce `lucas_kanade.max_level` from 5 to 3 — this cuts pyramid computation at the cost of handling smaller maximum displacements per frame.

---

## Known Limitations

- **No scale estimation** — bounding box dimensions are fixed at initialisation. Objects changing apparent size due to camera zoom or perspective shift will have an increasingly misaligned bbox.
- **Single-object only** — one `ObjectTracker` instance per run. Multi-vehicle tracking is not supported in the current architecture.
- **Manual initialisation required** — a human must draw the bounding box on the start frame. There is no automatic detection stage.
- **CPU-only throughput** — ~1.3–1.6 fps on 4K input. Not suitable for real-time deployment without resolution downscaling or GPU-accelerated optical flow.
- **Fixed occlusion budget** — velocity prediction sustains tracking for a maximum of 45 frames (~1.5 s at 30 fps). Longer full occlusions cause tracker deactivation.
- **Axis-aligned bounding box** — no rotation support. Vehicles turning at intersections will have bbox misalignment proportional to the turn angle.

---

## Academic Context

Developed as the final project for the Fundamentals of Computer Vision course.

| Field | Detail |
|-------|--------|
| University | National University of Computer & Emerging Sciences (FAST-NUCES) |
| Campus | Lahore |
| Department | Data Science & Artificial Intelligence |
| Course | Fundamentals of Computer Vision |
| Semester | Spring 2026 |
| Instructor | Mubasher Baig |
| Student | Ali Ahmad — Roll No. 23L-2619 |

The full technical report — covering algorithm design decisions, parameter analysis, and per-test evaluation — is available in [`report/cv_project_tracking_report.pdf`](report/cv_project_tracking_report.pdf).

---

## Citation

If you use this project in your research or coursework, please cite it as:

```bibtex
@misc{ahmad2026harrislk,
  author       = {Ahmad, Ali},
  title        = {Harris--LK Vehicle Tracker: Feature-Based Single-Object Tracking
                  using Harris Corners and Lucas--Kanade Optical Flow},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/whozahm3d/harris-lk-vehicle-tracker}}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full details.

---

<p align="center">
  <a href="https://github.com/whozahm3d">Ali Ahmad</a> · FAST-NUCES Lahore · Spring 2026
</p>
