# Migration Guide

This guide helps you upgrade from previous versions to the latest version with activity detection and structured outputs.

## Overview

The analysis pipeline has been significantly enhanced with automatic activity detection and a reorganized folder structure. This migration guide covers the changes and required actions.

## Before/After Folder Structure

### Previous Structure (v1.x)

```
night_2025-12-24/
├── frames/                   # Input frames
├── sky_mask.png              # Generated mask
├── persistent_edges.png      # Generated mask
├── combined_mask.png         # Generated mask
├── metrics.csv               # Output metrics
├── events.json               # Output events
├── plots/                    # Visualizations
├── annotated/                # Annotated frames
├── timelapse_raw.mp4         # Timelapse (if generated)
└── timelapse_annotated.mp4   # Annotated timelapse (if generated)
```

### New Structure (v2.x)

```
night_2025-12-24/
├── frames/                   # Input frames (unchanged)
├── masks/                    # NEW: Organized mask directory
│   ├── sky_mask.png
│   ├── persistent_edges.png
│   └── combined_mask.png
├── data/                     # NEW: Structured data directory
│   ├── metrics.csv           # Enhanced with new columns
│   ├── events.json
│   └── activity_windows.json # NEW: Activity detection output
├── activity/                 # NEW: Per-window artifacts
│   ├── window_00_0045_0089/
│   │   ├── timelapse_window.mp4
│   │   ├── keogram.png
│   │   └── startrails.png
│   └── window_01_0234_0267/
│       ├── timelapse_window.mp4
│       ├── keogram.png
│       └── startrails.png
├── plots/                    # Visualizations (unchanged location)
├── annotated/                # Annotated frames (unchanged location)
└── timelapse/                # NEW: Timelapse video directory
    └── timelapse_annotated.mp4
```

## Path Migration Table

| Old Path | New Path | Notes |
|----------|----------|-------|
| `metrics.csv` | `data/metrics.csv` | Enhanced with `focus_score`, `interest_score`, `z_streak` columns |
| `events.json` | `data/events.json` | Same format, new location |
| `sky_mask.png` | `masks/sky_mask.png` | Same content, organized location |
| `persistent_edges.png` | `masks/persistent_edges.png` | Same content, organized location |
| `combined_mask.png` | `masks/combined_mask.png` | Same content, organized location |
| `timelapse_annotated.mp4` | `timelapse/timelapse_annotated.mp4` | Organized location |
| N/A | `data/activity_windows.json` | **NEW** - Activity window metadata |
| N/A | `activity/window_XX_YYYY_ZZZZ/` | **NEW** - Per-window artifacts |

## Action Required for Users

### 1. Update Scripts Reading Output Files

If you have scripts that read analysis outputs, update the paths:

**Python example:**
```python
# OLD
metrics_path = night_dir / "metrics.csv"
events_path = night_dir / "events.json"

# NEW
metrics_path = night_dir / "data" / "metrics.csv"
events_path = night_dir / "data" / "events.json"

# NEW - Access activity windows
windows_path = night_dir / "data" / "activity_windows.json"
```

**Bash example:**
```bash
# OLD
csvtool col 3 night_2025-12-24/metrics.csv

# NEW
csvtool col 3 night_2025-12-24/data/metrics.csv
```

### 2. Update metrics.csv Column References

The CSV now includes additional columns:

**Previous columns:**
- `file`
- `mean_brightness`
- `star_contrast`
- `streak_count`

**New columns (in addition to above):**
- `focus_score` - Image sharpness metric (Laplacian variance)
- `interest_score` - Weighted activity score for frame
- `z_streak` - Robust z-score of streak count

**Example: Reading with pandas**
```python
import pandas as pd

df = pd.read_csv("night_2025-12-24/data/metrics.csv")

# New columns available
print(df.columns)
# ['file', 'mean_brightness', 'star_contrast', 'streak_count', 
#  'focus_score', 'interest_score', 'z_streak']

# Plot interest score over time
import matplotlib.pyplot as plt
plt.plot(df['interest_score'])
plt.xlabel('Frame')
plt.ylabel('Interest Score')
plt.title('Activity Interest Over Time')
plt.show()
```

### 3. No Changes Required For

The following remain **unchanged** and require no migration:

- Input frame location: `frames/` directory
- Plot outputs: `plots/` directory
- Annotated frames: `annotated/` directory
- Configuration file: `config.yaml`
- Command line interface: `analyze.py` commands work the same

### 4. Re-run Analysis

**Important:** The new version automatically organizes outputs. Simply re-run analysis on existing night directories:

```bash
cd analysis_simple
python analyze.py ../data/night_2025-12-24/ --all-tools
```

This will:
- Create new directory structure
- Generate all new outputs (activity windows, per-window artifacts)
- Preserve your input frames (unchanged)

## New Features Overview

### Activity Window Detection

The pipeline now automatically identifies periods of high activity using an **interest scoring algorithm**:

```python
interest_score = (
    0.60 × robust_zscore(streak_count) +
    0.15 × robust_zscore(|Δbrightness|) +
    0.15 × robust_zscore(|Δcontrast|) +
    0.10 × robust_zscore(|Δfocus|)
)
```

Activity windows are detected when `interest_score >= threshold` (default: 3.5) for at least `min_len` frames (default: 6).

**Output:** `data/activity_windows.json`

```json
[
  {
    "start": 45,
    "end": 89,
    "peak_index": 67,
    "peak_value": 8.42,
    "length": 45
  }
]
```

### Per-Window Artifacts

For each detected activity window, the pipeline generates:

1. **Timelapse video** (`timelapse_window.mp4`) - Short video of just the activity period
2. **Keogram** (`keogram.png`) - Vertical center slice stacked over time (shows motion)
3. **Startrails** (`startrails.png`) - Maximum projection showing all trails

These enable rapid visual review of potentially interesting periods without watching full-night timelapses.

### Focus Metrics

New `focus_score` measurement tracks image sharpness:
- High values = sharp stars, good focus
- Low values = blur, clouds, defocus
- Useful for data quality assessment and filtering

## Configuration Changes

New optional sections in `config.yaml`:

```yaml
# Activity window detection (optional - uses defaults if omitted)
windows:
  threshold: 3.5       # Interest score threshold for detection
  min_len: 6           # Minimum frames for a valid window
  pad: 10              # Padding frames around detected activity
  merge_gap: 8         # Merge windows closer than this
  max_windows: 10      # Maximum windows to report
  w_streak: 0.60       # Weight for streak count in interest score
  w_mean: 0.15         # Weight for brightness change
  w_contrast: 0.15     # Weight for contrast change
  w_focus: 0.10        # Weight for focus change

# Keogram generation (optional)
keogram:
  column_width: 3      # Width of center column to sample

# Startrail generation (optional)
startrails:
  gamma: 1.0           # Gamma correction (1.0 = none)
```

**Default values are used if these sections are omitted** - no action required unless you want to customize behavior.

## Using Activity Windows for ML Workflows

The new structured outputs are optimized for machine learning:

### 1. Extract High-Value Frames for Labeling

```python
import json
from pathlib import Path

# Load activity windows
with open("night_2025-12-24/data/activity_windows.json") as f:
    windows = json.load(f)

# Extract frames from top 3 activity windows
for window in windows[:3]:
    start, end = window["start"], window["end"]
    print(f"Review frames {start}-{end} (peak: {window['peak_value']:.2f})")
    
    # Copy frames to labeling directory
    for i in range(start, end + 1):
        frame_path = Path(f"night_2025-12-24/frames/frame_{i:03d}.jpg")
        # ... copy to labeling tool
```

### 2. Feature Engineering

```python
import pandas as pd

df = pd.read_csv("night_2025-12-24/data/metrics.csv")

# Use interest_score for classification
df['has_activity'] = df['interest_score'] > 3.5

# Use focus_score for quality filtering
df_good_quality = df[df['focus_score'] > df['focus_score'].quantile(0.25)]

# Use z_streak for anomaly detection
df_anomalies = df[df['z_streak'] > 3.0]
```

### 3. Training Data Generation

```python
# Create balanced dataset using activity windows
import numpy as np

# Load windows
with open("night_2025-12-24/data/activity_windows.json") as f:
    windows = json.load(f)

# Positive samples (frames within windows)
positive_frames = []
for w in windows:
    positive_frames.extend(range(w['start'], w['end'] + 1))

# Negative samples (frames outside windows)
all_frames = set(range(len(df)))
negative_frames = list(all_frames - set(positive_frames))

# Sample balanced dataset
num_samples = min(len(positive_frames), len(negative_frames))
train_frames = (
    np.random.choice(positive_frames, num_samples, replace=False).tolist() +
    np.random.choice(negative_frames, num_samples, replace=False).tolist()
)
```

## Troubleshooting

### "File not found" errors in custom scripts

Update your scripts to use the new paths:
- `metrics.csv` → `data/metrics.csv`
- `events.json` → `data/events.json`

### Want to keep old directory structure

The new structure is automatic and cannot be disabled. However, you can create symbolic links:

```bash
cd night_2025-12-24
ln -s data/metrics.csv metrics.csv
ln -s data/events.json events.json
```

### No activity windows detected

This is normal for nights with little transient activity. Adjust configuration:

```yaml
windows:
  threshold: 2.5  # Lower threshold (default: 3.5)
  min_len: 4      # Shorter minimum window (default: 6)
```

### Too many activity windows detected

Increase the threshold:

```yaml
windows:
  threshold: 4.5  # Higher threshold (default: 3.5)
```

## Support

For questions or issues:
1. Check updated documentation in `analysis_simple/README.md`
2. Review example outputs in `data/night_2025-12-24/`
3. Open an issue on GitHub

## Summary

**Key Changes:**
- ✅ Outputs now organized in subdirectories (masks/, data/, activity/, timelapse/)
- ✅ New activity detection with interest scoring
- ✅ Per-window artifacts for efficient review
- ✅ Enhanced metrics.csv with focus and interest scores
- ✅ ML-ready structured outputs

**Action Required:**
- Update scripts to read from `data/` subdirectory
- Handle new columns in metrics.csv
- Re-run analysis to generate new outputs

**Backward Compatibility:**
- Input frames location unchanged
- Command line interface unchanged
- Existing config.yaml works (new sections are optional)
