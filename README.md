# 🚗 Harris–LK Vehicle Tracker

> Feature-Based Single-Object Tracking using Harris Corners and Lucas–Kanade Optical Flow

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=flat-square&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Status](https://img.shields.io/badge/Status-Academic%20Project-orange?style=flat-square)]()

A classical, interpretable single-object vehicle tracker built entirely with **Harris Corner Detection** and **pyramidal Lucas–Kanade optical flow** — no deep learning, no Kalman filter. Designed to track fast-moving vehicles in 4K drone and surveillance footage, with robust handling for uniform object texture, background competition, partial occlusion, and frame-exit degradation.

---

## 📋 Table of Contents

- [🎬 Demo](#-demo)
- [⚙️ How It Works](#️-how-it-works)
- [✨ Key Features](#-key-features)
- [📁 Project Structure](#-project-structure)
- [🛠️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [🔧 Configuration Reference](#-configuration-reference)
- [📊 Results](#-results)
- [💻 Tested Environment](#-tested-environment)
- [🔍 Troubleshooting](#-troubleshooting)
- [⚠️ Known Limitations](#️-known-limitations)
- [🎓 Academic Context](#-academic-context)
- [📝 Citation](#-citation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🎬 Demo

<!-- Add tracked output GIF here -->
> 📌 GIF of tracked vehicle will be added here. See `results/` folder after running the pipeline.

---

## ⚙️ How It Works

The tracker follows a strict 6-step pipeline per frame, matching the classical feature-based tracking specification:

| Step | Description |
|------|-------------|
| **1** | Read video frame |
| **2** | Detect Harris corners inside the user-defined bounding box on the start frame |
| **3** | Track corners using pyramidal LK optical flow (forward pass) |
| **4** | Filter unreliable points via forward-backward error check (round-trip error < 3.5 px) |
| **5** | Update bounding box via median displacement shift of surviving points |
| **6** | Repeat for all frames — with velocity prediction and survival-based redetection as fallbacks |

**Additional mechanisms beyond the base pipeline:**
- **Velocity-guided background filter** — when object speed > 3 px/frame, discards points moving < 30% of estimated velocity (eliminates stationary road corners)
- **Survival-based redetection** — triggers Harris redetection when surviving point count drops below 25% of initial count
- **Velocity prediction** — blends measured displacement with exponentially smoothed velocity estimate when point count drops critically low; falls back to pure velocity when zero points survive
- **CLAHE preprocessing** — applied before both Harris detection and LK tracking for illumination robustness

---

## ✨ Key Features

- 🔍 **Harris corner detection** with CLAHE preprocessing for illumination robustness
- 🌊 **Pyramidal LK optical flow** with forward-backward error filtering
- 🚫 **Velocity-guided background filter** to reject stationary road and background corners
- 🔄 **Survival-based redetection** — triggers only when genuinely needed, not on a fixed schedule
- 📐 **Velocity prediction** for smooth tracking during object exit and brief occlusion
- ⚙️ **Config-driven design** — only 3 lines change between test videos (`source`, `start_frame`, `bbox`)
- 🎥 **H.264 MP4 output** with bounding box, centroid trail, and corner overlay rendered on every frame
- 📂 **Timestamped results folders** — successive runs never overwrite each other

---

## 📁 Project Structure

```
harris-lk-vehicle-tracker/
├── core/
│   ├── harris_detector.py        # Harris corner detector with CLAHE and mask
│   ├── lucas_kanade.py           # LK tracker with forward-backward error filtering
│   └── object_tracker.py         # Stateful single-object tracker with velocity prediction
├── utils/
│   ├── config_loader.py          # YAML config with dot-access attributes
│   ├── folder_manager.py         # Timestamped output directory manager
│   ├── health_check.py           # Pre-flight environment validation
│   └── resolution_utils.py       # Frame resolution helpers
├── metrics/
│   ├── logger.py                 # Levelled logging (INFO / DEBUG / WARNING)
│   └── performance_tracker.py    # Per-frame metric accumulation and export
├── visualization/
│   ├── display.py                # Bounding box and trail overlay rendering
│   └── plotter.py                # Result plot generation and saving
├── video_io/
│   ├── output_writer.py          # Video writer wrapper
│   └── video_source.py           # Threaded video source with buffering
├── notebooks/
│   ├── phase1_infrastructure.ipynb   # Config, logging, folder management
│   ├── phase2_algorithms.ipynb       # Harris detection + interactive bbox selection
│   ├── phase3_tracking.ipynb         # LK optical flow + FB filtering
│   ├── phase4_results.ipynb          # Evaluation, metrics, plots
│   └── phase5_integration.ipynb      # Full end-to-end pipeline
├── results/                      # ⚠️ Auto-generated — excluded from repo (see .gitignore)
├── video/                        # ⚠️ Place your test videos here — excluded from repo
├── report/
│   └── cv_project_tracking_report.pdf
├── main.py                       # CLI entry point
├── config.yaml                   # All tunable parameters
├── requirements.txt              # Python dependencies
├── CONTRIBUTING.md               # Contribution guidelines
└── LICENSE                       # MIT License
```

> **Note:** The `results/` and `video/` directories are excluded from this repository due to large file sizes. Create the `video/` folder manually and place your test videos inside it before running.

---

## 🛠️ Installation

### Prerequisites

- Python 3.12
- Windows 11 (recommended — see [Troubleshooting](#-troubleshooting) for other platforms)
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

Download `openh264-1.8.0-win64.dll` from the [Cisco OpenH264 releases page](https://github.com/cisco/openh264/releases) and place it in the **project root directory** alongside `main.py`. This is required for H.264 MP4 output via OpenCV on Windows.

**5. Create the video folder**

```bash
mkdir video
```

Place your test video(s) inside the `video/` folder.

---

## 🚀 Usage

### Step 1 — Configure `config.yaml`

Open `config.yaml` and set the three input fields for your video:

```yaml
input:
  source: "C:\\path\\to\\your\\video\\test_1.mp4"   # Full path to video file
  start_frame: 0                                      # Frame index to start tracking
  bbox: [x, y, width, height]                         # Bounding box in full resolution pixels
```

> **Windows path tip:** Use double backslashes (`\\`) or forward slashes (`/`) in the path string.

### Step 2 — Select frame and draw bounding box

Run `notebooks/phase2_algorithms.ipynb` in Jupyter. This notebook will:
- Open a scrubber window to navigate through the video and select the start frame
- Launch `cv2.selectROI` for you to draw the bounding box around the target vehicle
- Print the exact `start_frame` and `bbox` values to paste into `config.yaml`

### Step 3 — Run the full tracking pipeline

Run `notebooks/phase5_integration.ipynb` in Jupyter. This will:
- Load config, initialize the tracker, and process all frames
- Display a live preview window during tracking
- Save the annotated output video to `results/<video_name>_<timestamp>/videos/`
- Save run logs, metrics JSON, and plots to the same timestamped folder

### CLI Usage (alternative)

You can also run the tracker directly from the command line without Jupyter:

```bash
python main.py --config config.yaml
```

### Output Structure

Each run produces a self-contained timestamped folder:

```
results/
└── test_1_20260516_173936/
    ├── videos/
    │   └── test_1_tracked.mp4     # Annotated output video
    ├── logs/
    │   └── run.log                # Full run log with per-frame point counts
    ├── plots/
    │   ├── phase_2/               # Harris corner visualizations
    │   ├── phase_3/               # LK flow and bbox update plots
    │   ├── phase_4/               # Point survival and centroid trajectory
    │   └── integration_snapshots.png
    ├── metrics/
    │   └── test_1_metrics.json    # Avg, min, max points and frame count
    └── params/
        └── test_1_params.json     # Config snapshot for this run
```

---

## 🔧 Configuration Reference

<details>
<summary>Click to expand full configuration reference</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Input** | | |
| `input.source` | `""` | Full path to the input video file |
| `input.start_frame` | `0` | Frame index to begin tracking from |
| `input.bbox` | `[x, y, w, h]` | Bounding box in full-resolution pixels (set via Phase 2) |
| **Harris Corner Detection** | | |
| `harris.max_corners` | `400` | Maximum number of Harris corners to detect inside bbox |
| `harris.quality_level` | `0.005` | Minimum corner quality relative to the strongest corner (lower = more corners) |
| `harris.min_distance` | `3` | Minimum Euclidean distance in pixels between detected corners |
| `harris.block_size` | `5` | Neighbourhood size for computing corner response |
| `harris.k` | `0.04` | Harris sensitivity constant (standard value) |
| `harris.redetect_threshold` | `0.25` | Survival ratio below which redetection triggers (0.25 = below 25% of initial count) |
| **Lucas–Kanade Optical Flow** | | |
| `lucas_kanade.win_size` | `[41, 41]` | LK search window size in pixels — larger handles faster motion |
| `lucas_kanade.max_level` | `5` | Number of image pyramid levels — higher handles larger displacements |
| `lucas_kanade.max_iter` | `30` | Maximum iterations for LK convergence |
| `lucas_kanade.epsilon` | `0.01` | Convergence criterion for LK iteration |
| `lucas_kanade.min_eig_threshold` | `0.001` | Minimum eigenvalue threshold — discards poorly conditioned patches |
| `lucas_kanade.fb_error_threshold` | `3.5` | Forward-backward round-trip error threshold in pixels — points above this are rejected |
| **Tracking Logic** | | |
| `tracking.min_points_per_object` | `4` | Minimum surviving points required for a reliable median displacement estimate |
| `tracking.bbox_smoothing` | `0.0` | Temporal smoothing factor for bbox updates (0.0 = no smoothing, snap to median exactly) |
| `tracking.redetect_interval` | `60` | Reserved scheduled redetection interval (currently unused — survival-based only) |
| **Memory** | | |
| `memory.max_trail_length` | `60` | Maximum number of centroid positions stored in the trail deque |
| `memory.max_inactive_frames` | `45` | Frames of zero-point prediction allowed before tracker deactivates (~1.5s at 30fps) |
| **Preprocessing** | | |
| `preprocessing.clahe_enable` | `true` | Enable CLAHE contrast normalisation before Harris detection and LK tracking |
| `preprocessing.clahe_clip_limit` | `3.5` | CLAHE clip limit — controls contrast enhancement strength |
| `preprocessing.clahe_tile_size` | `[4, 4]` | CLAHE tile grid size for local histogram equalisation |
| **Output** | | |
| `output.save_video` | `true` | Save annotated tracking video to results folder |
| `output.show_fps` | `true` | Display FPS overlay on the output video |
| `output.show_trails` | `true` | Render centroid trail on the output video |

</details>

---

## 📊 Results

The system was evaluated on two 4K test sequences with 100% tracker stability on both.

| Metric | 🚁 Test 1 (Drone, 24 fps) | 📷 Test 2 (Surveillance, 30 fps) |
|--------|--------------------------|----------------------------------|
| Resolution | 3840 × 2160 | 3840 × 2160 |
| Total Frames | 280 | 529 |
| Initial Corners | 67 | 137 |
| Avg Points / Frame | 42.0 | 137.0 |
| Tracker Stability | ✅ 100% | ✅ 100% |
| Processing Rate | 1.63 fps | 1.25 fps |
| Primary Challenge | Background competition | Exit-phase degradation |
| Key Fix Applied | Velocity-guided BG filter | Velocity prediction + 5% exit threshold |

**Test 1** — Top-down drone footage of a silver car on a road. The uniform silver car roof produces sparse Harris corners (~67 at init), which compete with strong road-marking corners. The velocity-guided background filter suppresses stationary road points and stabilises tracking.

**Test 2** — Oblique surveillance footage of a car exiting frame-left. The oblique angle exposes the car's side panels, windows, and wheel arches, yielding 137 stable corners throughout all 529 frames. Exit-phase handling via velocity prediction keeps the bbox following the car to the frame boundary.

> **Hardware:** All results obtained on CPU only — no GPU acceleration used.

---

## 💻 Tested Environment

| Component | Version |
|-----------|---------|
| OS | Windows 11 |
| Python | 3.12 |
| OpenCV | 4.9.0.80 |
| NumPy | 1.26.4 |
| Jupyter | Notebook / JupyterLab |
| GPU | Not required |

---

## 🔍 Troubleshooting

**❌ `Failed to initialize VideoWriter` or blank output video**

The OpenH264 DLL is missing from the project root.

```
Solution: Download openh264-1.8.0-win64.dll from
https://github.com/cisco/openh264/releases
and place it in the project root directory alongside main.py.
```

---

**❌ `Cannot open video: path/to/video.mp4`**

The video path in `config.yaml` is incorrect or the file does not exist.

```yaml
# ✅ Correct — double backslashes on Windows
source: "C:\\Users\\yourname\\harris-lk-vehicle-tracker\\video\\test_1.mp4"

# ✅ Also correct — forward slashes work too
source: "C:/Users/yourname/harris-lk-vehicle-tracker/video/test_1.mp4"

# ❌ Wrong — single backslashes will fail
source: "C:\Users\yourname\video\test_1.mp4"
```

---

**❌ `AttributeError: module 'ctypes' has no attribute 'windll'`**

You are running on Linux or macOS. The DPI awareness line in `main.py` is Windows-only.

```python
# In main.py, wrap the ctypes line as follows:
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass
```

---

## ⚠️ Known Limitations

- **No scale estimation** — the bounding box size is fixed at initialisation; objects changing apparent size will drift out of alignment
- **Single object only** — no multi-object support; only one `ObjectTracker` instance is maintained
- **Manual bounding box required** — the tracker must be initialised with a manually drawn bbox at the start frame
- **CPU-only throughput** — processes 4K footage at ~1.3–1.6 fps; not suitable for real-time deployment without resolution downscaling or GPU acceleration
- **Full occlusion timeout** — velocity prediction sustains tracking for at most 45 frames (~1.5 s); longer full occlusions cause deactivation
- **Fixed axis-aligned bounding box** — no rotation support; vehicles turning at intersections will have misaligned bboxes
- **Windows-specific DPI fix** — `ctypes.windll` line in `main.py` requires wrapping on non-Windows platforms

---

## 🎓 Academic Context

This project was developed as a final group project for the **Fundamentals of Computer Vision** course.

| Field | Detail |
|-------|--------|
| University | National University of Computer & Emerging Sciences |
| Campus | FAST NUCES — Lahore Campus |
| Department | Data Science & Artificial Intelligence |
| Course | Fundamentals of Computer Vision |
| Semester | Spring 2026 |
| Instructor | Mubasher Baig |
| Student | Ali Ahmad (Roll No. 23L-2619) |

---

## 📝 Citation

If you use this project in your research or coursework, please cite it as:

```bibtex
@misc{ahmad2026harrislk,
  author       = {Ahmad, Ali},
  title        = {Harris--LK Vehicle Tracker: Feature-Based Object Tracking
                  using Harris Corners and Lucas--Kanade Optical Flow},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/whozahm3d/harris-lk-vehicle-tracker}}
}
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full details.

---

<p align="center">Made with ❤️ by <a href="https://github.com/whozahm3d">Ali Ahmad</a> — FAST NUCES Lahore, Spring 2026</p>
