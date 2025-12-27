# astroplate-analysis

A collection of experimental tools for astronomical plate/time-lapse analysis to detect satellites, meteors, and other transient events.

## Repository Structure

This repository contains multiple analysis tools, each in its own self-contained folder with its own dependencies, configuration, and documentation.

### Analysis Tools

#### `analysis_simple/`
Basic computer vision pipeline using OpenCV for streak detection in time-lapse astronomical images.

- **Purpose**: Detect satellite streaks, meteors, and transient events using edge detection and Hough transforms
- **Dependencies**: OpenCV, NumPy, Matplotlib, PyYAML
- **Status**: Active development (v1)

See [`analysis_simple/README.md`](analysis_simple/README.md) for detailed usage and configuration.

#### Future Analysis Tools (Planned)
- `analysis_ml/` - Machine learning-based detection using neural networks
- `analysis_sk/` - scikit-learn based statistical analysis
- Additional experimental pipelines as they're developed

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

2. **Run Analysis**:
   ```bash
   cd ../../analysis_simple
   pip install -r requirements.txt
   python analyze.py ../data/night_2025-12-24/ --all-tools
   ```

   This generates:
   - **Masks** (sky_mask.png, persistent_edges.png, combined_mask.png)
   - **Metrics & Events** (data/metrics.csv, data/events.json, data/activity_windows.json)
   - **Activity Windows** (per-window timelapses, keograms, startrails in activity/ subfolder)
   - **Visualizations** (plots of brightness, contrast, and streak counts)
   - **Annotated Frames** (detected streaks overlaid on images)
   - **Timelapse Videos** (MP4 videos in timelapse/ subfolder)

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
│   └── activity_windows.json # Detected activity windows with ML-ready metadata
├── activity/                 # Per-window artifacts for high-interest periods
│   └── window_00_0123_0145/
│       ├── timelapse_window.mp4
│       ├── keogram.png
│       └── startrails.png
├── plots/                    # Time-series visualizations
├── annotated/                # Frames with detected streaks overlaid
└── timelapse/                # Full-night timelapse videos
    └── timelapse_annotated.mp4
```

### Activity Detection System

The pipeline automatically identifies periods of high astronomical activity using a weighted **interest scoring algorithm**:

- **Interest Score** = 0.60×(streak_count) + 0.15×(brightness_change) + 0.15×(contrast_change) + 0.10×(focus_change)
- Robust z-scores (MAD-based) eliminate sensitivity to outliers
- Configurable threshold, padding, and merging parameters
- Generates focused artifacts (timelapses, keograms, startrails) for each activity window

This enables efficient review of long observation sessions by automatically highlighting frames with potential meteors, satellites, or other transient events.

### Configuration Guide

Activity detection parameters can be tuned in `config.yaml`:

```yaml
windows:
  threshold: 3.5       # Interest score threshold for detection
  min_len: 6           # Minimum frames for a valid window
  pad: 10              # Padding frames around detected activity
  merge_gap: 8         # Merge windows closer than this
  max_windows: 10      # Maximum windows to report
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
