# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-12-27

### Added

- **Activity Window Detection**: Automatic identification of high-interest time periods using weighted interest scoring algorithm
  - Interest score combines streak count (60%), brightness change (15%), contrast change (15%), and focus change (10%)
  - Robust z-score calculation using MAD (Median Absolute Deviation) for outlier resistance
  - Configurable detection parameters: threshold, minimum length, padding, merge gap
  - Signal processing pipeline: compute metrics → calculate changes → robust normalization → window detection → merge nearby windows

- **Per-Window Artifacts**: Automated generation of focused outputs for each activity window
  - Timelapse videos (`timelapse_window.mp4`) for each detected activity period
  - Keograms (`keogram.png`) - vertical center slice stacked over time
  - Startrails (`startrails.png`) - per-pixel maximum projection with optional gamma correction
  - Stored in structured subdirectories: `activity/window_XX_YYYY_ZZZZ/`

- **Focus Metrics**: New image sharpness measurement using Laplacian variance
  - Added `focus_score` column to metrics.csv for tracking optical clarity and star sharpness
  - Used as a component in interest scoring to detect potential events

- **Interest Scoring**: Per-frame interest scores for ML and automated review
  - Added `interest_score` column to metrics.csv
  - Added `z_streak` column (robust z-score of streak count)
  - Enables automated identification of frames worth manual review

- **Structured Output Folders**: Organized output hierarchy for better data management
  - `masks/` - Analysis masks (sky_mask.png, persistent_edges.png, combined_mask.png)
  - `data/` - Structured data files (metrics.csv, events.json, activity_windows.json)
  - `activity/` - Per-window artifacts with chronological naming
  - `timelapse/` - Full-night timelapse videos
  - `plots/` - Time-series visualizations (existing, unchanged location)
  - `annotated/` - Annotated frames (existing, unchanged location)

- **activity_windows.json**: New output file documenting detected activity periods
  - Contains window metadata: start/end frame indices, length, peak index, peak interest score
  - JSON format for easy parsing in ML pipelines and data analysis tools
  - Sorted chronologically for sequential processing

- **Helper Functions**: New utility functions for signal processing and artifact generation
  - `focus_score()` - Compute Laplacian variance for focus measurement
  - `robust_zscore()` - MAD-based z-score calculation resistant to outliers
  - `detect_activity_windows()` - Signal processing pipeline for window detection
  - `build_keogram()` - Generate keogram from frame sequence
  - `build_startrails()` - Generate startrail composite with optional gamma
  - `ensure_dir()` - Safe directory creation
  - `load_gray_frame()` - Load and validate grayscale frames
  - `build_timelapse_from_list()` - Build timelapse from frame list

### Changed

- **File Locations**: Reorganized outputs into logical subdirectories
  - `metrics.csv` → `data/metrics.csv`
  - `events.json` → `data/events.json`
  - `sky_mask.png` → `masks/sky_mask.png`
  - `persistent_edges.png` → `masks/persistent_edges.png`
  - `combined_mask.png` → `masks/combined_mask.png`
  - Timelapse videos now in `timelapse/` subfolder

- **metrics.csv Schema**: Extended with new columns
  - Added `focus_score` - Laplacian variance measurement
  - Added `interest_score` - Weighted combination of activity signals
  - Added `z_streak` - Robust z-score of streak count
  - Retained existing columns: file, mean_brightness, star_contrast, streak_count

- **Tool Scripts**: Updated to read from new data/ subfolder
  - `tools/visualize.py` - Reads from `data/metrics.csv`
  - `tools/overlay_streaks.py` - Reads from `data/events.json`
  - `tools/validate_data.py` - Checks for `data/metrics.csv` and `data/events.json`

- **Command Line Help**: Updated to reflect new output structure in `analyze.py --help`

### Configuration

New configuration sections in `config.yaml`:

- **windows**: Activity window detection parameters
  ```yaml
  windows:
    threshold: 3.5       # Interest score threshold
    min_len: 6           # Minimum frames for window
    pad: 10              # Padding around activity
    merge_gap: 8         # Merge windows closer than this
    max_windows: 10      # Maximum windows to report
    w_streak: 0.60       # Weight for streak count
    w_mean: 0.15         # Weight for brightness change
    w_contrast: 0.15     # Weight for contrast change
    w_focus: 0.10        # Weight for focus change
  ```

- **keogram**: Keogram generation settings
  ```yaml
  keogram:
    column_width: 3      # Width of center column to sample
  ```

- **startrails**: Startrail generation settings
  ```yaml
  startrails:
    gamma: 1.0           # Gamma correction (1.0 = none)
  ```

### Developer Notes

- Activity detection uses a multi-stage pipeline: metric computation → change detection → robust z-scoring → window identification → merging → artifact generation
- Robust statistics (MAD-based z-scores) ensure outlier resistance without manual threshold tuning
- Per-window artifacts enable efficient review of long observation sessions
- Structured outputs are optimized for ML workflows and automated analysis
- Backward compatibility maintained for plots/ and annotated/ directories

### Migration

See [MIGRATION.md](MIGRATION.md) for upgrading from previous versions.
