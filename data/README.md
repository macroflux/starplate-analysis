# Data Directory

This directory contains astronomical time-lapse data used by analysis tools in this repository. Each subdirectory represents data from a single observation night.

## Structure

Each night directory follows this structure:

```
data/
├── night_2025-12-24/
│   ├── frames/              # INPUT: Place your .jpg frames here
│   │   ├── frame_001.jpg
│   │   ├── frame_002.jpg
│   │   └── ...
│   ├── masks/               # OUTPUT: Analysis masks
│   │   ├── sky_mask.png
│   │   ├── persistent_edges.png
│   │   └── combined_mask.png
│   ├── data/                # OUTPUT: Structured data files
│   │   ├── metrics.csv      # Per-frame statistics with focus_score and interest_score
│   │   ├── events.json      # Detected transient events
│   │   └── activity_windows.json  # Detected activity periods (NEW)
│   ├── activity/            # OUTPUT: Per-window artifacts (NEW)
│   │   ├── window_00_0045_0089/
│   │   │   ├── timelapse_window.mp4
│   │   │   ├── keogram.png
│   │   │   └── startrails.png
│   │   └── window_01_0234_0267/
│   │       ├── timelapse_window.mp4
│   │       ├── keogram.png
│   │       └── startrails.png
│   ├── plots/               # OUTPUT: Visualization plots
│   │   ├── brightness_over_time.png
│   │   ├── contrast_over_time.png
│   │   └── streak_counts.png
│   ├── annotated/           # OUTPUT: Frames with detected streaks overlaid
│   │   ├── frame_002.jpg
│   │   └── ...
│   └── timelapse/           # OUTPUT: Full-night timelapse videos (NEW)
│       └── timelapse_annotated.mp4
└── night_YYYY-MM-DD/        # Additional observation nights
    └── ...
```

### New in v2.0

- **masks/** - Organized directory for analysis masks (sky_mask.png, persistent_edges.png, combined_mask.png)
- **data/** - Structured data outputs optimized for ML workflows
  - `metrics.csv` - Now includes `focus_score`, `interest_score`, and `z_streak` columns
  - `activity_windows.json` - Automatically detected high-interest time periods
- **activity/** - Per-window artifacts for each detected activity period
  - Timelapse videos of just the activity window
  - Keograms showing motion over time
  - Startrail composites
- **timelapse/** - Organized location for full-night timelapse videos

## Usage

### Option 1: Download Images Automatically

Use the data_fetch tool to automatically download and organize images:

```bash
cd tools/data_fetch
python fetch.py YYYYMMDD
```

This creates `data/night_YYYY-MM-DD/frames/` and downloads all images.

### Option 2: Manual Setup

1. Create a directory named `night_YYYY-MM-DD` where YYYY-MM-DD is the observation date
2. Create a `frames/` subdirectory within it
3. Place your time-lapse `.jpg` images in the `frames/` directory

### Running Analysis

Once you have images in place, run the analysis:

```bash
cd analysis_simple/
python analyze.py ../data/night_YYYY-MM-DD/
```

### Image Requirements

- **Format**: JPEG with `.jpg` extension
- **Naming**: Any naming scheme works (files are sorted alphabetically)
- **Content**: Night sky images from a fixed camera position
- **Sequence**: Should be a time-series (sequential frames)

## Example Data

See `night_2025-12-24/` for an example night directory with sample outputs.

## Common Location

This `data/` directory serves as a common location where all analysis tools can:
- Read input frames (downloaded by `tools/data_fetch/` or placed manually)
- Generate output files (masks, metrics, events)
- Store visualization plots
- Share data between different analysis tools

Each analysis tool in the repository can reference data directories here using relative paths like `../data/night_YYYY-MM-DD/`.

## Git Ignore

The `night_*/` directories are git-ignored to avoid committing large image datasets. Only the directory structure and documentation are tracked in version control.
