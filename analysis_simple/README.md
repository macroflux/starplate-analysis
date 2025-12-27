# analysis_simple

Basic computer vision pipeline for detecting satellites, meteors, and other transient events in astronomical time-lapse images.

## Overview

This project provides tools to analyze sequences of astronomical images (time-lapse frames) to detect and catalog transient events such as:
- Satellite streaks
- Meteor trails
- Aircraft paths
- Other moving objects in the night sky

The analysis pipeline processes JPEG frames, builds a sky mask to exclude non-sky regions, and uses computer vision techniques (edge detection, Hough transforms) to identify linear streaks indicative of transient events.

## Features

- **Configurable Analysis Pipeline**: All detection parameters can be customized via YAML configuration
- **Smart Sky Masking**: Multi-frame analysis automatically excludes camera housing, overlay text, and dark regions
- **Persistent Edge Filtering**: Identifies and excludes static structures (trees, buildings, wires) that create false detections
- **Automatic Outlier Detection**: Filters out thumbnail images or frames with unusual dimensions
- **Streak Detection**: Uses Canny edge detection and Hough line transforms
- **Metrics Tracking**: Records brightness, contrast, and streak counts per frame
- **Event Cataloging**: Identifies and logs frames with significant transient activity
- **Visualization Tools**: Plot metrics over time and overlay detected streaks on images
- **Timelapse Generation**: Create MP4 videos from raw or annotated frame sequences
- **Validation Utilities**: Check data structure and image quality before analysis
- **Portable**: Automatically adapts to different camera locations and orientations

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `opencv-python` - Image processing, computer vision, and video encoding
- `numpy` - Numerical operations
- `matplotlib` - Plotting and visualization
- `PyYAML` - Configuration file parsing

## Usage

### Getting Data

If you don't have data yet, use the data_fetch tool to download images:

```bash
cd tools/data_fetch
python fetch.py 20251224
```

This automatically creates `../data/night_2025-12-24/frames/` and downloads all images.

### Running Analysis

#### Basic Analysis

Analyze a night's worth of frames:

```bash
python analysis_simple/analyze.py data/night_2025-12-24/
```

#### Run All Tools (Recommended)

Run validation, analysis, visualization, and overlay generation in one command:

```bash
python analysis_simple/analyze.py data/night_2025-12-24/ --all-tools
```

This will:
1. **Validate** data structure and frames before analysis
2. **Analyze** all frames and detect transient events
3. **Visualize** metrics with time-series plots
4. **Overlay** detected streaks on event frames
5. **Generate** timelapse videos from raw and annotated frames

#### Individual Tool Options

Run specific tools:

```bash
# Validate before analysis
python analysis_simple/analyze.py data/night_2025-12-24/ --validate

# Generate plots after analysis
python analysis_simple/analyze.py data/night_2025-12-24/ --visualize

# Create annotated frames with detected streaks
python analysis_simple/analyze.py data/night_2025-12-24/ --overlay

# Generate timelapse videos
python analysis_simple/analyze.py data/night_2025-12-24/ --timelapse

# Combine multiple options
python analysis_simple/analyze.py data/night_2025-12-24/ --validate --visualize --timelapse
```

#### With Custom Configuration

```bash
python analysis_simple/analyze.py data/night_2025-12-24/ --config custom_config.yaml --all-tools
```

#### Get Help

```bash
python analysis_simple/analyze.py --help
```

## Folder Structure

Your data should be organized as follows:

```
night_2025-12-24/
├── frames/
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   ├── frame_003.jpg
│   └── ...
├── config.yaml           (optional, night-specific configuration)
├── masks/                (generated each run)
│   ├── sky_mask.png
│   ├── persistent_edges.png
│   └── combined_mask.png
├── data/                 (generated output)
│   ├── metrics.csv       # Per-frame metrics with focus_score and interest_score
│   ├── events.json       # Detected transient events
│   └── activity_windows.json  # Detected activity periods (NEW)
├── activity/             (generated with activity detection)
│   ├── window_00_0045_0089/
│   │   ├── timelapse_window.mp4
│   │   ├── keogram.png
│   │   └── startrails.png
│   └── window_01_0234_0267/
│       ├── timelapse_window.mp4
│       ├── keogram.png
│       └── startrails.png
├── plots/                (generated with --visualize)
│   ├── brightness_over_time.png
│   ├── contrast_over_time.png
│   └── streak_counts.png
├── annotated/            (generated with --overlay)
│   ├── frame_002.jpg     (streak count shown in top right)
│   └── ...
└── timelapse/            (generated with --timelapse or --all-tools)
    └── timelapse_annotated.mp4  (if --overlay also used)
```

**Required:**
- `frames/` directory containing `.jpg` image files

**Optional:**
- `config.yaml` - Night-specific configuration (checked first before global config)

**Generated by analysis (regenerated each run):**
- `masks/sky_mask.png` - Binary mask excluding non-sky regions (dark obstructions)
- `masks/persistent_edges.png` - Mask excluding static structures (trees, buildings, wires)
- `masks/combined_mask.png` - Final analysis mask combining sky_mask and persistent_edges
- `data/metrics.csv` - Per-frame statistics with focus and interest scores
- `data/events.json` - Detected transient events
- `data/activity_windows.json` - Detected high-interest activity periods (NEW)

**Generated per activity window:**
- `activity/window_XX_YYYY_ZZZZ/timelapse_window.mp4` - Timelapse of activity window
- `activity/window_XX_YYYY_ZZZZ/keogram.png` - Vertical center slice over time
- `activity/window_XX_YYYY_ZZZZ/startrails.png` - Maximum projection of window frames

**Generated with --visualize:**
- `plots/` - Directory containing time-series plots

**Generated with --overlay:**
- `annotated/` - Directory containing frames with detected streaks drawn

**Generated with --timelapse:**
- `timelapse/timelapse_annotated.mp4` - MP4 video timelapse (if --overlay also used)

## Output Files

### data/metrics.csv

CSV file with one row per frame containing:

| Column | Description |
|--------|-------------|
| `file` | Filename of the frame |
| `mean_brightness` | Average pixel brightness in sky region |
| `star_contrast` | Standard deviation of high-pass filtered image (star visibility metric) |
| `streak_count` | Number of linear streaks detected |
| `focus_score` | Laplacian variance (image sharpness metric) - NEW |
| `interest_score` | Weighted activity score for ML and filtering - NEW |
| `z_streak` | Robust z-score of streak count - NEW |

Example:
```csv
file,mean_brightness,star_contrast,streak_count,focus_score,interest_score,z_streak
frame_001.jpg,45.23,12.8,0,234.5,-0.82,-1.15
frame_002.jpg,44.89,13.1,3,245.1,6.42,4.23
frame_003.jpg,46.12,12.5,1,238.7,0.91,0.58
```

The new columns enable:
- **focus_score**: Quality filtering (higher = sharper stars, better focus)
- **interest_score**: Automated high-value frame identification (used for activity windows)
- **z_streak**: Normalized streak count for statistical analysis

### data/events.json

JSON file containing detected events (frames with significant streak activity):

```json
[
  {
    "file": "frame_002.jpg",
    "reason": "many_streaks",
    "streaks": [
      [120, 50, 180, 200],
      [250, 80, 280, 220]
    ]
  }
]
```

Each streak is represented as `[x1, y1, x2, y2]` - the start and end coordinates of the detected line.

### data/activity_windows.json (NEW)

JSON file containing automatically detected activity periods with high interest scores:

```json
[
  {
    "start": 45,
    "end": 89,
    "peak_index": 67,
    "peak_value": 8.42,
    "length": 45
  },
  {
    "start": 234,
    "end": 267,
    "peak_index": 251,
    "peak_value": 6.85,
    "length": 34
  }
]
```

Fields:
- `start` - Starting frame index (0-based)
- `end` - Ending frame index (inclusive)
- `peak_index` - Frame with highest interest score in window
- `peak_value` - Maximum interest score in window
- `length` - Number of frames in window (end - start + 1)

Windows are sorted chronologically and limited to `max_windows` (default: 10).

### masks/sky_mask.png

Grayscale image where:
- White pixels (255) = sky region to analyze
- Black pixels (0) = excluded regions (housing, overlays, dark areas)

Built using low-percentile analysis across 60 sampled frames to identify persistent dark obstructions.

### masks/persistent_edges.png

Grayscale image identifying static structures:
- White pixels (255) = persistent edges to exclude (trees, buildings, wires, panel seams)
- Black pixels (0) = areas where edges don't repeat consistently

Built by detecting edges across 80 sampled frames. Edges that appear in ≥20% of samples are considered "persistent" and excluded from streak detection. This dramatically reduces false positives from treelines and static structures while preserving detection of real transient events.

**Why this works:** Trees, buildings, and fixed structures create edges in the same locations frame after frame. Real sky events (meteors, satellites, planes) do not repeat in the same pixels.

### masks/combined_mask.png

Grayscale image showing the final analysis region:
- White pixels (255) = active analysis region where streaks are detected
- Black pixels (0) = excluded regions (combined exclusions from sky_mask and persistent_edges)

This mask is the result of combining `sky_mask.png` with `persistent_edges.png`:
```
combined_mask = sky_mask AND NOT persistent_edges
```

**Purpose:** This visualization shows you exactly what region of each frame is being analyzed for transient events. Review this file to verify that:
- Important sky regions are included (white)
- False positive sources are excluded (black)
- The balance between sky coverage and noise reduction is appropriate

**Using this for configuration tuning:**
- If too much sky is excluded (too much black), increase `persistent_edges.keep_fraction` in your config
- If false positives persist, decrease `persistent_edges.keep_fraction` or increase `persistent_edges.dilate_px`
- Adjust settings and re-run analysis to see updated mask - masks regenerate each run

### activity/window_XX_YYYY_ZZZZ/ (NEW)

Per-window artifact directories are automatically generated for each detected activity window:
- Directory naming: `window_<index>_<start_frame>_<end_frame>`
- Example: `window_00_0045_0089` = first window, frames 45-89

**Contents:**

1. **timelapse_window.mp4** - Short timelapse video of just this activity period
   - Enables rapid review without watching full-night footage
   - Same FPS as configured in `timelapse.fps` setting

2. **keogram.png** - Vertical center slice stacked horizontally over time
   - Shows motion patterns across the sky
   - Useful for identifying satellite passes and meteor directions
   - Width controlled by `keogram.column_width` setting

3. **startrails.png** - Per-pixel maximum projection of all frames in window
   - Shows accumulated star trails and transient paths
   - Optional gamma correction via `startrails.gamma` setting
   - Reveals patterns not visible in individual frames

## Configuration

The analysis pipeline can be customized using a YAML configuration file.

### Configuration File Search Priority

The script searches for configuration in this order:
1. **Explicit path** via `--config` flag (highest priority)
2. **Night directory** - `data/night_2025-12-24/config.yaml` (night-specific settings)
3. **Script directory** - `analysis_simple/config.yaml` (global defaults)

This allows you to have global defaults while overriding settings for specific nights.

### Configuration Parameters

```yaml
masking:
  sample_count: 60              # Number of frames to sample for mask building
  low_percentile: 20            # Percentile for identifying dark obstructions
  obstruction_threshold: 35     # Pixel brightness threshold for obstructions
  dilate_px: 8                  # Dilation pixels for safety margin
  overlay_region:               # Region to mask out (e.g., camera overlay text)
    top: 0
    bottom: 140
    left: 0
    right: 450
  resize_width: 0               # 0 = no resize (keep full resolution)

persistent_edges:
  sample_count: 80              # Number of frames to sample for edge detection
  canny_low: 40                 # Lower threshold for Canny edge detection
  canny_high: 120               # Upper threshold for Canny edge detection
  hp_sigma: 2.0                 # High-pass filter sigma (reduces clouds/glow)
  keep_fraction: 0.20           # Edge must appear in >=20% of samples
  dilate_px: 6                  # Dilation for tree sway tolerance

analysis:
  gaussian_sigma: 5             # Sigma for Gaussian blur in star contrast calculation

streak_detection:
  canny_low: 40                 # Lower threshold for Canny edge detection
  canny_high: 120               # Upper threshold for Canny edge detection
  hough_threshold: 60           # Accumulator threshold for Hough line detection
  min_line_length: 40           # Minimum line length in pixels
  max_line_gap: 10              # Maximum gap between line segments to connect

events:
  min_streaks: 2                # Minimum streaks to flag as an event

timelapse:
  fps: 30                       # Frames per second for video output
  quality: 8                    # Video quality 0-10 (lower = higher quality)

# Activity window detection (NEW)
windows:
  threshold: 3.5       # Interest score threshold for activity detection
  min_len: 6           # Minimum frames for a window
  pad: 10              # Padding frames around detected activity
  merge_gap: 8         # Merge windows closer than this many frames
  max_windows: 10      # Maximum number of windows to report
  w_streak: 0.60       # Weight for streak count in interest score
  w_mean: 0.15         # Weight for brightness change
  w_contrast: 0.15     # Weight for contrast change
  w_focus: 0.10        # Weight for focus change

# Keogram generation (NEW)
keogram:
  column_width: 3      # Width of center column to sample for keogram

# Startrails generation (NEW)
startrails:
  gamma: 1.0           # Gamma correction for startrails (1.0 = none)
```

### Creating Custom Configuration

**Global configuration:**
```bash
cd analysis_simple/
# Edit config.yaml for global defaults
```

**Night-specific configuration:**
```bash
cp analysis_simple/config.yaml data/night_2025-12-24/config.yaml
# Edit data/night_2025-12-24/config.yaml for this night only
```

**Explicit configuration:**
```bash
python analysis_simple/analyze.py data/night_2025-12-24/ --config path/to/custom_config.yaml
```

### Key Configuration Tips

**Reducing false positives from trees:**
- Lower `persistent_edges.keep_fraction` (try 0.15-0.25)
- Increase `persistent_edges.dilate_px` (6-10 for tree sway)
- Increase `streak_detection.min_line_length` (80-150)

**More sensitive detection:**
- Lower `streak_detection.hough_threshold`
- Lower `streak_detection.min_line_length`

**Masks regenerate on every run** using current configuration, so you can experiment freely.

## How Persistent Edge Masking Works

One of the key challenges in detecting transient sky events is distinguishing real events (satellites, meteors, aircraft) from static features in the environment like trees, buildings, and wires.

### The Problem

Traditional edge detection treats every edge equally, so:
- Treelines create hundreds of edge segments
- Building rooflines generate long lines
- Guy-wires and power lines appear as streaks
- These false positives overwhelm real sky events

### The Solution: Persistent Edge Detection

The system now uses a two-stage masking approach:

**Stage 1: Sky Mask**
- Samples 60 frames across the night
- Uses low-percentile analysis to identify dark obstructions
- Excludes: camera housing, overlay text, consistently dark regions

**Stage 2: Persistent Edge Mask**
- Samples 80 frames across the night
- Runs edge detection on each frame (within sky mask only)
- Accumulates edges: counts how many times each pixel is detected as an edge
- Creates "persistent edge" mask: pixels that are edges in ≥20% of frames
- Dilates slightly to account for tree sway and camera shake

**Final Combined Mask:**
```
effective_analysis_region = sky_mask AND NOT persistent_edges
```

### Why This Works

**Static structures** create edges in the same locations frame after frame:
- Trees: same branches appear in same pixels
- Buildings: roofline edges persist
- Wires: consistent linear features

**Real sky events** never repeat in the same pixels:
- Satellites move across different parts of sky
- Meteors appear in random locations
- Aircraft paths vary

### Benefits

✅ **Portable**: Automatically adapts when you relocate camera  
✅ **No manual annotation**: No need to draw exclusion regions  
✅ **Reduces false positives**: Typically 80-95% reduction in false detections  
✅ **Preserves real events**: Stars remain as points, sky events don't repeat spatially  
✅ **Self-updating**: Regenerates each run based on current frames

### Inspecting the Masks

After running analysis, you can open the generated mask files to verify:
- `sky_mask.png` - Should show the sky region in white
- `persistent_edges.png` - Should highlight treelines, buildings, fixed structures in white
- `combined_mask.png` - Shows the final analysis region (white = analyzed, black = excluded)

**Review `combined_mask.png` to understand what's being analyzed:**
- This is the actual mask used for streak detection
- White regions are where transient events will be detected
- Black regions are completely excluded from analysis

If `persistent_edges.png` shows too much sky highlighted, adjust `persistent_edges.keep_fraction` to a higher value (0.25-0.30).

## Activity Detection Algorithm

The analysis pipeline includes an automatic activity window detection system that identifies periods of high astronomical interest. This enables efficient review of long observation sessions by highlighting frames with potential meteors, satellites, or other transient events.

### Overview

Activity detection uses a multi-stage signal processing pipeline:

1. **Metric Computation** - Calculate per-frame measurements (brightness, contrast, focus, streaks)
2. **Change Detection** - Compute first differences to identify abrupt changes
3. **Robust Normalization** - Apply MAD-based z-scores resistant to outliers
4. **Interest Scoring** - Weighted combination of normalized signals
5. **Window Detection** - Identify contiguous regions above threshold
6. **Window Merging** - Combine nearby windows to reduce fragmentation
7. **Artifact Generation** - Create per-window timelapses, keograms, and startrails

### Interest Score Formula

The interest score combines multiple signals weighted by their importance:

```
interest_score = 
    0.60 × robust_zscore(streak_count) +
    0.15 × robust_zscore(|Δbrightness|) +
    0.15 × robust_zscore(|Δcontrast|) +
    0.10 × robust_zscore(|Δfocus|)
```

**Weights (configurable):**
- **w_streak (0.60)**: Streak count is the strongest signal for transient events
- **w_mean (0.15)**: Brightness changes may indicate clouds, aircraft lights, or dawn/dusk
- **w_contrast (0.15)**: Contrast changes may indicate atmospheric conditions affecting star visibility
- **w_focus (0.10)**: Focus changes may indicate optical issues or environmental factors

**Why these weights?**
- Streaks are direct evidence of transient events (satellites, meteors)
- Changes in brightness/contrast/focus provide contextual information
- Weights can be tuned for different observation goals (e.g., increase w_focus for optical quality monitoring)

### Robust Z-Score Calculation

Traditional z-scores are sensitive to outliers. The pipeline uses **MAD-based robust z-scores**:

```python
def robust_zscore(x):
    median = np.median(x)
    mad = np.median(|x - median|)
    scale = 1.4826 * mad  # Convert MAD to approximate standard deviation
    return (x - median) / scale
```

**Benefits:**
- **Outlier resistant**: Single extreme values don't skew the baseline
- **No manual thresholds**: Adapts to each night's data distribution
- **Statistically sound**: MAD is a robust estimator of scale
- **Comparable across nights**: Z-scores have consistent interpretation

### Window Detection Process

**Step 1: Identify Active Regions**
- Mark all frames where `interest_score >= threshold` (default: 3.5)
- This creates binary "active" / "inactive" labels

**Step 2: Find Contiguous Runs**
- Group consecutive active frames into runs
- Example: Frames [45, 46, 47, 48] form one run

**Step 3: Filter by Minimum Length**
- Discard runs shorter than `min_len` (default: 6 frames)
- Prevents flagging momentary noise spikes

**Step 4: Add Padding**
- Extend each run by `pad` frames (default: 10) on both sides
- Provides context around detected activity
- Example: Run [45-48] → Window [35-58]

**Step 5: Merge Nearby Windows**
- If two windows are within `merge_gap` (default: 8) frames, combine them
- Reduces fragmentation of continuous activity periods
- Example: Windows [35-58] and [62-89] → Merged [35-89]

**Step 6: Cap Total Windows**
- Keep only top `max_windows` (default: 10) by peak interest score
- Ensures manageable output even for very active nights

**Step 7: Sort Chronologically**
- Final windows are ordered by start frame for sequential processing

### Signal Processing Details

**Change Detection:**
```python
Δbrightness = diff1(mean_brightness)  # First difference
Δcontrast = diff1(star_contrast)
Δfocus = diff1(focus_score)
```

First differences highlight abrupt changes (e.g., sudden appearance of bright satellite).

**Absolute Values:**
Interest score uses `|Δ|` (absolute change) because both increases and decreases may indicate activity.

**Prepending First Value:**
`diff1()` prepends the first value to maintain array length, ensuring all frames have an interest score.

### Artifact Generation

For each detected window, the pipeline generates:

**1. Timelapse Video** (`timelapse_window.mp4`)
- Short video of just the activity period
- Same FPS as full-night timelapse
- Enables rapid review without scrubbing through hours of footage

**2. Keogram** (`keogram.png`)
- Vertical center slice stacked horizontally (time on x-axis)
- Shows motion patterns across the sky
- Useful for identifying satellite passes and meteor trajectories
- Height = frame height, Width = number of frames in window

**3. Startrails** (`startrails.png`)
- Per-pixel maximum projection over all frames in window
- Shows accumulated star trails and transient paths
- Optional gamma correction for visibility enhancement
- Reveals patterns not visible in individual frames

### Configuration Tuning

**Adjust threshold for sensitivity:**
```yaml
windows:
  threshold: 3.5  # Higher = fewer windows, lower = more windows
```

- **threshold: 2.5** - More sensitive, catches subtle activity
- **threshold: 4.5** - Less sensitive, only major events
- **threshold: 3.5** - Default balanced setting

**Adjust window characteristics:**
```yaml
windows:
  min_len: 6      # Shorter = catch brief events
  pad: 10         # Larger = more context around activity
  merge_gap: 8    # Larger = fewer fragmented windows
```

**Adjust signal weights:**
```yaml
windows:
  w_streak: 0.80    # Emphasize streak detection
  w_mean: 0.10      # De-emphasize brightness changes
  w_contrast: 0.05
  w_focus: 0.05
```

### Example Workflow

1. **Run analysis**: `python analyze.py night_2025-12-24/ --all-tools`
2. **Review windows**: Check `data/activity_windows.json` for detected periods
3. **Watch timelapses**: Review `activity/window_XX_YYYY_ZZZZ/timelapse_window.mp4` files
4. **Inspect keograms**: Look for diagonal streaks indicating motion
5. **Verify startrails**: Check for unusual patterns in accumulated trails
6. **Extract frames**: Use window start/end indices to locate frames for detailed analysis

## Machine Learning Integration

The structured outputs are designed for machine learning workflows, enabling automated dataset generation and training data preparation.

### 1. Training Data Extraction

Use `activity_windows.json` to automatically extract high-value frames for labeling:

```python
import json
from pathlib import Path
import shutil

# Load activity windows
with open("night_2025-12-24/data/activity_windows.json") as f:
    windows = json.load(f)

# Extract frames from top 3 activity windows for labeling
labeling_dir = Path("ml_dataset/to_label")
labeling_dir.mkdir(parents=True, exist_ok=True)

for i, window in enumerate(windows[:3]):
    start, end = window["start"], window["end"]
    print(f"Window {i}: frames {start}-{end}, peak={window['peak_value']:.2f}")
    
    # Copy frames to labeling directory
    for frame_idx in range(start, end + 1):
        src = Path(f"night_2025-12-24/frames/frame_{frame_idx:03d}.jpg")
        dst = labeling_dir / f"window{i:02d}_frame_{frame_idx:04d}.jpg"
        if src.exists():
            shutil.copy(src, dst)
```

**Benefits:**
- Automated high-value frame selection
- Reduces manual review time by 80-90%
- Focuses labeling effort on frames with events

### 2. Feature Engineering

Use `metrics.csv` for ML features and filtering:

```python
import pandas as pd
import numpy as np

# Load metrics
df = pd.read_csv("night_2025-12-24/data/metrics.csv")

# Binary classification target
df['has_activity'] = df['interest_score'] > 3.5

# Quality filtering
df_good = df[df['focus_score'] > df['focus_score'].quantile(0.25)]

# Feature matrix for ML
features = df[['mean_brightness', 'star_contrast', 'focus_score', 'streak_count']].values
labels = df['has_activity'].values

# Temporal features (rolling statistics)
df['brightness_rolling_mean'] = df['mean_brightness'].rolling(window=5, center=True).mean()
df['streak_rolling_max'] = df['streak_count'].rolling(window=5, center=True).max()
```

**Available features:**
- `mean_brightness` - Sky brightness level
- `star_contrast` - Star visibility measure
- `focus_score` - Image sharpness
- `streak_count` - Number of detected lines
- `interest_score` - Pre-computed activity score (can be used as target or feature)
- `z_streak` - Normalized streak count

### 3. Balanced Dataset Generation

Create balanced positive/negative samples using activity windows:

```python
# Load windows and metrics
with open("night_2025-12-24/data/activity_windows.json") as f:
    windows = json.load(f)
df = pd.read_csv("night_2025-12-24/data/metrics.csv")

# Positive samples (frames within windows)
positive_indices = []
for w in windows:
    positive_indices.extend(range(w['start'], w['end'] + 1))

# Negative samples (frames outside windows)
all_indices = set(range(len(df)))
negative_indices = list(all_indices - set(positive_indices))

# Create balanced dataset
n_samples = min(len(positive_indices), len(negative_indices))
pos_sample = np.random.choice(positive_indices, n_samples, replace=False)
neg_sample = np.random.choice(negative_indices, n_samples, replace=False)

balanced_indices = np.concatenate([pos_sample, neg_sample])
balanced_labels = np.array([1]*n_samples + [0]*n_samples)

# Extract frames
for idx, label in zip(balanced_indices, balanced_labels):
    frame_file = df.iloc[idx]['file']
    src = Path(f"night_2025-12-24/frames/{frame_file}")
    dst = Path(f"ml_dataset/train/{label}/{frame_file}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
```

### 4. Multi-Night Dataset Aggregation

Combine multiple nights for larger training sets:

```python
import pandas as pd
from pathlib import Path

all_metrics = []
all_windows = []

for night_dir in Path("data").glob("night_*"):
    # Load metrics
    metrics_path = night_dir / "data" / "metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df['night'] = night_dir.name
        all_metrics.append(df)
    
    # Load windows
    windows_path = night_dir / "data" / "activity_windows.json"
    if windows_path.exists():
        with open(windows_path) as f:
            windows = json.load(f)
            for w in windows:
                w['night'] = night_dir.name
            all_windows.extend(windows)

# Combined dataframe
df_all = pd.concat(all_metrics, ignore_index=True)
print(f"Total frames: {len(df_all)}")
print(f"Total activity windows: {len(all_windows)}")

# Distribution analysis
print("\nActivity distribution:")
print(df_all['interest_score'].describe())
```

### 5. Event Classification

Use `events.json` for bounding box supervision:

```python
import json

# Load events
with open("night_2025-12-24/data/events.json") as f:
    events = json.load(f)

# Convert to YOLO format
for event in events:
    frame_file = event['file']
    streaks = event['streaks']  # [[x1, y1, x2, y2], ...]
    
    # Convert to center-x, center-y, width, height (normalized)
    labels = []
    for x1, y1, x2, y2 in streaks:
        cx = (x1 + x2) / 2 / img_width
        cy = (y1 + y2) / 2 / img_height
        w = abs(x2 - x1) / img_width
        h = abs(y2 - y1) / img_height
        labels.append(f"0 {cx} {cy} {w} {h}")  # class 0 = streak
    
    # Save YOLO label file
    label_file = f"ml_dataset/labels/{frame_file.replace('.jpg', '.txt')}"
    Path(label_file).write_text("\n".join(labels))
```

### 6. Temporal Sequence Models

Activity windows enable sequence-based learning:

```python
# Create sequences from activity windows
sequences = []
sequence_labels = []

for window in windows:
    start, end = window['start'], window['end']
    
    # Extract feature sequence
    sequence = df.iloc[start:end+1][['mean_brightness', 'star_contrast', 
                                       'focus_score', 'streak_count']].values
    sequences.append(sequence)
    
    # Label: peak interest score or binary activity
    sequence_labels.append(window['peak_value'] > 5.0)

# Pad sequences to same length
from tensorflow.keras.preprocessing.sequence import pad_sequences
X = pad_sequences(sequences, dtype='float32', padding='post')
y = np.array(sequence_labels)

# Train LSTM/GRU model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### 7. Semi-Supervised Learning

Use interest scores for pseudo-labeling:

```python
# High-confidence positive samples
high_activity = df[df['interest_score'] > 6.0]

# High-confidence negative samples
low_activity = df[df['interest_score'] < -1.0]

# Uncertain samples for active learning
uncertain = df[(df['interest_score'] >= -1.0) & (df['interest_score'] <= 1.0)]

print(f"High confidence positives: {len(high_activity)}")
print(f"High confidence negatives: {len(low_activity)}")
print(f"Uncertain (need labeling): {len(uncertain)}")
```

### 8. Visualization for Validation

Use per-window artifacts for visual validation:

```python
# Review keograms for motion patterns
import cv2
import matplotlib.pyplot as plt

for i, window in enumerate(windows[:5]):
    keogram_path = f"night_2025-12-24/activity/window_{i:02d}_{window['start']:04d}_{window['end']:04d}/keogram.png"
    keogram = cv2.imread(keogram_path, cv2.IMREAD_GRAYSCALE)
    
    plt.figure(figsize=(12, 4))
    plt.imshow(keogram, cmap='gray', aspect='auto')
    plt.title(f"Window {i}: Frames {window['start']}-{window['end']}")
    plt.xlabel("Time (frames)")
    plt.ylabel("Vertical Position")
    plt.colorbar(label="Brightness")
    plt.savefig(f"ml_dataset/validation/keogram_window{i:02d}.png")
    plt.close()
```

### Benefits for ML Workflows

✅ **Automated data curation** - No manual frame review needed  
✅ **Balanced sampling** - Easy positive/negative split using windows  
✅ **Rich features** - Pre-computed metrics ready for training  
✅ **Temporal context** - Window structure preserves time-series information  
✅ **Visual validation** - Keograms and startrails aid verification  
✅ **Scalable** - Process multiple nights automatically  
✅ **Flexible** - Supports classification, detection, and sequence models

## Visualization

The visualization and overlay tools are now integrated into `analyze.py` via command-line flags. However, they can still be run independently if needed.

### Plot Metrics Over Time

Visualize brightness, contrast, and streak counts:

**Integrated method (recommended):**
```bash
python analysis_simple/analyze.py data/night_2025-12-24/ --visualize
```

**Standalone method:**
```bash
python analysis_simple/tools/visualize.py data/night_2025-12-24/
```

Generates plots in `data/night_2025-12-24/plots/`:
- `brightness_over_time.png`
- `contrast_over_time.png`
- `streak_counts.png`

Optional flags:
- `--show` - Display plots interactively
- `--output <dir>` - Specify custom output directory

### Overlay Detected Streaks

Draw detected streaks on the original images:

**Integrated method (recommended):**
```bash
python analysis_simple/analyze.py data/night_2025-12-24/ --overlay
```

**Standalone method:**
```bash
python analysis_simple/tools/overlay_streaks.py data/night_2025-12-24/
```

Creates annotated images in `data/night_2025-12-24/annotated/` with detected streaks drawn in red.

Optional flags:
- `--output <dir>` - Specify custom output directory

### Generate Timelapse Videos

Create MP4 timelapse videos from frame sequences:

**Integrated method (recommended):**
```bash
# Generate both raw and annotated timelapses (if --overlay also used)
python analysis_simple/analyze.py data/night_2025-12-24/ --timelapse --overlay

# Or just raw frames
python analysis_simple/analyze.py data/night_2025-12-24/ --timelapse
```

**Standalone method:**
```bash
# Timelapse from raw frames
python analysis_simple/tools/timelapse.py data/night_2025-12-24/frames/

# Timelapse from annotated frames
python analysis_simple/tools/timelapse.py data/night_2025-12-24/annotated/ --output annotated.mp4

# Custom settings
python analysis_simple/tools/timelapse.py data/night_2025-12-24/frames/ \
    --output night.mp4 --fps 25 --quality 5
```

Generates:
- `timelapse_raw.mp4` - Video of all raw frames (integrated mode)
- `timelapse_annotated.mp4` - Video of annotated frames (if --overlay used)

Optional flags (standalone):
- `--output <path>` - Output MP4 file path
- `--fps <int>` - Frames per second (default: 30)
- `--quality <0-10>` - Video quality, lower=higher (default: 8)
- `--pattern <glob>` - Frame file pattern (default: *.jpg)
- `--quiet` - Suppress progress messages

## Validation

Before running analysis, validate your data structure:

**Integrated method (recommended):**
```bash
python analysis_simple/analyze.py data/night_2025-12-24/ --validate
```

**Standalone method:**
```bash
python analysis_simple/tools/validate_data.py data/night_2025-12-24/
```

This checks:
- Directory structure is correct
- Frames directory exists and contains images
- Image format and dimensions are consistent
- Configuration file validity
- Warns about potential issues

## Example Data

See `../data/night_2025-12-24/` for a sample data structure with example outputs.

## Troubleshooting

### Too many false positives (treelines, structures)
- The persistent edge mask should handle this automatically
- If still seeing false positives, try:
  - Lower `persistent_edges.keep_fraction` (0.15-0.20)
  - Increase `persistent_edges.dilate_px` (8-12)
  - Increase `streak_detection.min_line_length` (80-150)
- Open `persistent_edges.png` to verify it's capturing the problem areas

### Too few detections / missing real events
- Increase `persistent_edges.keep_fraction` (0.25-0.30) to be less aggressive
- Lower `streak_detection.hough_threshold` for more sensitive detection
- Lower `streak_detection.min_line_length` to catch shorter streaks
- Check if `persistent_edges.png` is masking too much sky

### Frames with different dimensions / thumbnails
- The script automatically detects and filters outlier dimensions
- Check console output for "Filtered out X frame(s) with different dimensions"
- These are typically thumbnail files that can be safely ignored

### "No frames found"
- Ensure your images are in `.jpg` format (not `.jpeg`, `.png`, etc.)
- Check that images are in the `frames/` subdirectory
- Run validation: `python analysis_simple/analyze.py <night_dir> --validate`

### Config file priority questions
- The script checks in this order:
  1. `--config` explicit path (highest priority)
  2. `night_dir/config.yaml` (night-specific)
  3. `analysis_simple/config.yaml` (global defaults)
- Place night-specific configs in the data directory for automatic loading

### Masks not updating after config changes
- Masks are now regenerated on every run automatically
- Old mask files are deleted at the start of each analysis
- No need to manually delete mask files anymore

### "IndexError: boolean index did not match"
- This means some frames have different dimensions than others
- The script now automatically skips frames with mismatched dimensions
- Check warnings in the output for which frames were skipped
- Run validation to identify problematic frames: `--validate`

### Memory issues with large datasets
- Consider reducing `sample_count` values in configuration
- Process nights individually rather than batch processing

### Video generation issues
- Timelapse generation uses OpenCV's VideoWriter with H.264 codec (avc1)
- If videos won't play, your system may need codec support
- All dependencies are already included with opencv-python
- Monitor progress output to identify problematic frames

### Tool scripts not running
- Ensure you're running from the project root directory
- Check that tool scripts exist in `analysis_simple/tools/`
- Run tools individually to see detailed error messages

## Contributing

Contributions are welcome! This is an experimental project for exploring ML techniques in astronomical image analysis.

## License

See repository for license information.
