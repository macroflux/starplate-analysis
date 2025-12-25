# starplate-analysis

ML experiments in astronomical plate/time-lapse analysis for detecting satellites, meteors, and other transient events.

## Overview

This project provides tools to analyze sequences of astronomical images (time-lapse frames) to detect and catalog transient events such as:
- Satellite streaks
- Meteor trails
- Aircraft paths
- Other moving objects in the night sky

The analysis pipeline processes JPEG frames, builds a sky mask to exclude non-sky regions, and uses computer vision techniques (edge detection, Hough transforms) to identify linear streaks indicative of transient events.

## Features

- **Configurable Analysis Pipeline**: All detection parameters can be customized via YAML configuration
- **Sky Masking**: Automatically excludes camera housing, overlay text, and dark regions
- **Streak Detection**: Uses Canny edge detection and Hough line transforms
- **Metrics Tracking**: Records brightness, contrast, and streak counts per frame
- **Event Cataloging**: Identifies and logs frames with significant transient activity
- **Visualization Tools**: Plot metrics over time and overlay detected streaks on images
- **Validation Utilities**: Check data structure and image quality before analysis

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `opencv-python` - Image processing and computer vision
- `numpy` - Numerical operations
- `matplotlib` - Plotting and visualization
- `PyYAML` - Configuration file parsing

## Usage

### Basic Analysis

Analyze a night's worth of frames:

```bash
python prep1/analyze1.py ./night_2025-12-24
```

### With Custom Configuration

```bash
python prep1/analyze1.py ./night_2025-12-24 --config custom_config.yaml
```

### Get Help

```bash
python prep1/analyze1.py --help
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
├── sky_mask.png          (generated on first run)
├── metrics.csv           (generated output)
└── events.json           (generated output)
```

**Required:**
- `frames/` directory containing `.jpg` image files

**Generated:**
- `sky_mask.png` - Binary mask excluding non-sky regions
- `metrics.csv` - Per-frame statistics
- `events.json` - Detected transient events

## Output Files

### metrics.csv

CSV file with one row per frame containing:

| Column | Description |
|--------|-------------|
| `file` | Filename of the frame |
| `mean_brightness` | Average pixel brightness in sky region |
| `star_contrast` | Standard deviation of high-pass filtered image (star visibility metric) |
| `streak_count` | Number of linear streaks detected |

Example:
```csv
file,mean_brightness,star_contrast,streak_count
frame_001.jpg,45.23,12.8,0
frame_002.jpg,44.89,13.1,3
frame_003.jpg,46.12,12.5,1
```

### events.json

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

### sky_mask.png

Grayscale image where:
- White pixels (255) = sky region to analyze
- Black pixels (0) = excluded regions (housing, overlays, dark areas)

## Configuration

The analysis pipeline can be customized using a YAML configuration file. If no config is specified, the script looks for `config.yaml` in the current directory, or uses built-in defaults.

### Configuration Parameters

```yaml
masking:
  brightness_threshold: 10      # Pixels darker than this are excluded
  overlay_region:               # Region to mask out (e.g., camera overlay text)
    top: 0
    bottom: 140
    left: 0
    right: 450

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
```

### Creating Custom Configuration

1. Copy the default `config.yaml`:
   ```bash
   cp config.yaml my_config.yaml
   ```

2. Edit parameters as needed

3. Run analysis with custom config:
   ```bash
   python prep1/analyze1.py ./night_data --config my_config.yaml
   ```

## Visualization

### Plot Metrics Over Time

Visualize brightness, contrast, and streak counts:

```bash
python tools/visualize.py ./night_2025-12-24
```

Generates plots in `./night_2025-12-24/plots/`:
- `brightness_over_time.png`
- `contrast_over_time.png`
- `streak_counts.png`

### Overlay Detected Streaks

Draw detected streaks on the original images:

```bash
python tools/overlay_streaks.py ./night_2025-12-24
```

Creates annotated images in `./night_2025-12-24/annotated/` with detected streaks drawn in red.

## Validation

Before running analysis, validate your data structure:

```bash
python tools/validate_data.py ./night_2025-12-24
```

This checks:
- Directory structure is correct
- Frames directory exists and contains images
- Image format and dimensions
- Warns about potential issues

## Example Data

See `examples/night_2025-12-24/` for a sample data structure with example outputs.

## Troubleshooting

### "No frames found"
- Ensure your images are in `.jpg` format (not `.jpeg`, `.png`, etc.)
- Check that images are in the `frames/` subdirectory
- Run validation: `python tools/validate_data.py <night_dir>`

### "Config file not found"
- The script looks for `config.yaml` in the current directory by default
- Specify a custom path with `--config` flag
- The script will use defaults if no config is found

### Poor detection results
- Adjust `streak_detection` parameters in `config.yaml`
- Lower `hough_threshold` for more sensitive detection
- Increase `min_line_length` to filter out noise
- Check `sky_mask.png` to ensure the mask covers the right region

### Memory issues with large datasets
- Process nights in smaller batches
- Reduce image resolution before analysis
- Monitor progress output to identify problematic frames

## Contributing

Contributions are welcome! This is an experimental project for exploring ML techniques in astronomical image analysis.

## License

See repository for license information.
