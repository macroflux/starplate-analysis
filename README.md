# astroplate-analysis

Automated post-processing and machine learning pipeline for all-sky camera observations. Analyzes overnight image sequences from unattended all-sky systems (AllSky, auto-exposure rigs) to detect, classify, and catalog transient astronomical events.

**Purpose:** Beyond simple event detection, this pipeline builds structured datasets for feature engineering, trains classifiers to categorize transient types (satellites, meteors, aircraft), and performs in-depth analysis of detected events. Combines rule-based computer vision with machine learning to progressively improve detection accuracy and enable automated event classification.

**Workflow:** Capture images overnight → Batch analyze with dual detection (interest-based + ML-based) → Extract features and build training data → Refine classifiers → Automatically categorize and analyze events → Review concentrated activity windows with confidence scores.

## Repository Structure

This repository contains multiple analysis tools, each in its own self-contained folder with its own dependencies, configuration, and documentation.

### Analysis Tools

#### `analysis_simple/`
Basic computer vision pipeline using OpenCV for streak detection and interest-based activity window detection.

- **Purpose**: Detect satellite streaks, meteors, and transient events using edge detection, Hough transforms, and automated activity window detection
- **Key Features**: Focus scoring, interest scoring, per-window artifacts (timelapses, keograms, startrails)
- **Dependencies**: OpenCV, NumPy, Matplotlib, PyYAML
- **Status**: Active development (v2.0)

See [`analysis_simple/README.md`](analysis_simple/README.md) for detailed usage and configuration.

#### `analysis_ml_activity_classifier/`
Machine learning activity classifier using logistic regression with peak-seeded windowing for detecting activity periods from per-frame features.

- **Purpose**: Train classifier on pseudo-labeled data and detect ML-based activity windows using peak-seeded windowing algorithm
- **Key Features**: Peak detection, EMA smoothing, adaptive window boundaries, filename fields in outputs
- **Dependencies**: NumPy, scikit-learn patterns (custom implementation)
- **Status**: Active (v1.0)

See [`analysis_ml_activity_classifier/README.md`](analysis_ml_activity_classifier/README.md) for detailed usage.

#### `analysis_ml_windows/`
ML window detection tool that applies peak-seeded windowing to classifier probabilities for detecting activity periods.

- **Purpose**: Detect activity windows from ML predictions with configurable smoothing and thresholding
- **Key Features**: EMA/MA smoothing, per-window artifacts, configurable detection parameters
- **Dependencies**: NumPy, PyYAML, OpenCV (for artifacts)
- **Status**: Active (v1.0)

See [`analysis_ml_windows/README.md`](analysis_ml_windows/README.md) for detailed usage.

### Utility Tools

#### `tools/data_fetch/`
Robust data fetching tool for downloading astronomical images from allsky.local server.

- **Purpose**: Download astroplate images with retry logic, progress tracking, and error handling
- **Dependencies**: requests, BeautifulSoup4, PyYAML, tqdm
- **Status**: Active (v1)

See [`tools/data_fetch/README.md`](tools/data_fetch/README.md) for detailed usage and configuration.

See [`tools/README.md`](tools/README.md) for information about all utility tools.

## Getting Started

### Complete Workflow

1. **Download Images**:
   ```bash
   cd tools/data_fetch
   pip install -r requirements.txt
   python fetch.py 20251224
   ```

2. **Run Basic Analysis**:
   ```bash
   cd ../../analysis_simple
   pip install -r requirements.txt
   python analyze.py ../data/night_2025-12-24/ --all-tools
   ```

   This generates:
   - **Masks** (masks/sky_mask.png, persistent_edges.png, combined_mask.png)
   - **Metrics & Events** (data/metrics.csv, data/events.json, data/activity_windows.json)
   - **Activity Windows** (per-window timelapses, keograms, startrails in activity/ subfolder)
   - **Visualizations** (plots/ - brightness, contrast, streak counts)
   - **Annotated Frames** (annotated/ - detected streaks overlaid)
   - **Timelapse Videos** (timelapse/ - full-night MP4 videos)

3. **Choose Your Window Source**:

   The analyzer supports three window detection strategies via the `--windows-source` flag:

   **Interest-based (default)** - Fast, rule-based detection:
   ```bash
   python analyze.py ../data/night_2025-12-24/ --windows-source interest
   ```

   **ML-based** - Uses trained classifier (requires ML setup first):
   ```bash
   # First, train classifier and generate ML windows
   cd ../analysis_ml_activity_classifier
   pip install -r requirements-ml.txt
   python train.py ../data/night_2025-12-24/
   cd ../analysis_ml_windows
   python infer_windows.py ../data/night_2025-12-24/
   
   # Then run analysis with ML windows
   cd ../analysis_simple
   python analyze.py ../data/night_2025-12-24/ --windows-source ml --all-tools
   ```

   **Hybrid** - Best of both worlds (ML + non-overlapping interest windows):
   ```bash
   python analyze.py ../data/night_2025-12-24/ --windows-source hybrid --all-tools
   ```

   The hybrid mode:
   - Starts with ML windows (higher confidence)
   - Adds interest windows that don't significantly overlap (IoU < 0.3)
   - Merges overlapping windows
   - Captures both learned patterns and novel events

   This unified interface means **all window sources produce artifacts in the same location** (`activity/`), making downstream processing consistent regardless of detection method.

### Individual Tool Usage

Each tool is independent and should be run from within its own directory. See individual tool READMEs for detailed usage.

## Tool Development Guidelines

Each tool folder should contain:
- `README.md` - Complete usage documentation
- `requirements.txt` - Python dependencies
- `config.yaml` or similar - Configuration files
- `tools/` - Tool-specific utilities and helper scripts
- `.github/` (optional) - Tool-specific CI/CD workflows
- Source code files

Tools should be self-contained and independently runnable.

## Output Structure

The analysis pipeline generates a structured output hierarchy optimized for machine learning and data analysis:

```
night_2025-12-24/
├── frames/                   # INPUT: Raw image sequence
├── masks/                    # Generated analysis masks
│   ├── sky_mask.png
│   ├── persistent_edges.png
│   └── combined_mask.png
├── data/                     # Structured data outputs
│   ├── metrics.csv           # Per-frame metrics with focus_score and interest_score
│   ├── events.json           # Detected transient events with streak coordinates
│   ├── activity_windows.json # Interest-based activity windows
│   └── ml_windows.json       # ML-based activity windows (optional)
├── activity/                 # Interest-based per-window artifacts
│   └── window_00_0123_0145/
│       ├── timelapse_window.mp4
│       ├── keogram.png
│       └── startrails.png
├── activity_ml/              # ML-based per-window artifacts (optional)
│   └── window_00_0157_0178/
│       ├── timelapse_window.mp4
│       ├── timelapse_annotated_window.mp4
│       ├── keogram.png
│       └── startrails.png
├── ml/                       # ML classifier outputs (optional)
│   ├── predictions.csv       # Per-frame probabilities (raw + smoothed)
│   ├── predictions_smoothed.csv
│   ├── report.json           # Training metrics
│   └── topk_frames.json      # High-confidence frames
├── plots/                    # Time-series visualizations
│   ├── brightness_over_time.png
│   ├── contrast_over_time.png
│   └── streak_counts.png
├── annotated/                # Frames with detected streaks overlaid
│   └── frame_*.jpg
└── timelapse/                # Full-night timelapse videos
    ├── timelapse.mp4
    └── timelapse_annotated.mp4
```

### Activity Detection System

The pipeline offers **unified window-based activity detection** with three complementary strategies selectable via `--windows-source`:

#### 1. Interest-Based Detection (default)

Weighted **interest scoring algorithm** using per-frame metrics:

- **Interest Score** = 0.60×(streak_count) + 0.15×(brightness_change) + 0.15×(contrast_change) + 0.10×(focus_change)
- Robust z-scores (MAD-based) eliminate sensitivity to outliers
- Configurable threshold, padding, and merging parameters
- Fast, rule-based, requires no training
- **Output**: `data/activity_windows.json`, artifacts in `activity/`

#### 2. ML-Based Detection

**Peak-seeded windowing** on classifier probabilities:

- Trains logistic regression on pseudo-labeled frames (from interest windows)
- Uses 8 features: z-scored metrics + temporal deltas
- **Peak-seeded algorithm**: EMA smoothing → peak detection → adaptive boundary expansion → merging
- Handles short events (meteors) and intermittent activity (satellites through clouds)
- Data-driven, learns patterns, provides confidence scores
- Requires ML training setup (see workflow above)
- **Output**: `data/ml_windows.json`, artifacts in `activity/`

#### 3. Hybrid Mode

**Best of both worlds** - combines ML and interest-based detection:

- Starts with ML windows (typically higher confidence)
- Adds non-overlapping interest windows (IoU < 0.3 threshold)
- Merges any overlapping windows using existing merge logic
- Captures both learned patterns (ML) and novel events (interest)
- Configurable IoU threshold via `hybrid_iou_threshold` in config
- **Output**: `data/windows_hybrid.json`, artifacts in `activity/`

#### Unified Interface

All three strategies produce artifacts in the **same location** (`activity/`), making downstream processing consistent. The `--windows-source` flag provides a single, predictable interface that future dashboards can drive.

**Recommendation**: Start with interest-based for quick results. Add ML detection after accumulating data. Use hybrid mode for maximum coverage.

### Configuration Guide

Activity detection parameters can be tuned in `config.yaml`:

```yaml
windows:
  threshold: 3.5       # Interest score threshold for detection
  min_len: 6           # Minimum frames for a valid window
  pad: 10              # Padding frames around detected activity
  merge_gap: 8         # Merge windows closer than this
  max_windows: 10      # Maximum windows to report
  hybrid_iou_threshold: 0.3  # IoU threshold for hybrid mode (lower = more interest windows added)
  w_streak: 0.60       # Weight for streak count
  w_mean: 0.15         # Weight for brightness change
  w_contrast: 0.15     # Weight for contrast change
  w_focus: 0.10        # Weight for focus change

keogram:
  column_width: 3      # Width of center column to sample

startrails:
  gamma: 1.0           # Gamma correction (1.0 = none)
```

See [`analysis_simple/README.md`](analysis_simple/README.md) for comprehensive configuration documentation.

### Machine Learning Preparation

The structured outputs are designed for ML workflows:

1. **Training Data Generation**: Use `activity_windows.json` to extract high-value frames for labeling
2. **Feature Engineering**: `metrics.csv` contains per-frame features (brightness, contrast, focus_score, interest_score)
3. **Event Classification**: `events.json` provides labeled transient events with bounding coordinates
4. **Automated Annotation**: Per-window artifacts (keograms, startrails) aid visual validation
5. **Dataset Splitting**: Use window timestamps for chronological train/val/test splits

This eliminates manual frame review and creates ML-ready datasets from raw astronomical observations.

## Data

The `data/` directory contains astronomical observation data organized by night. The `tools/data_fetch/` utility automatically creates and populates these directories when downloading images. All analysis tools read from and write to this common data location.

See [`data/README.md`](data/README.md) for structure details.

## Contributing

This is an experimental repository. Each tool may be at different stages of development. Check individual tool READMEs for status and contribution guidelines.

## License

See repository for license information.
