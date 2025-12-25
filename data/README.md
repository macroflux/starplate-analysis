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
│   ├── sky_mask.png         # OUTPUT: Generated mask
│   ├── metrics.csv          # OUTPUT: Per-frame statistics
│   ├── events.json          # OUTPUT: Detected events
│   └── plots/               # OUTPUT: Visualization plots
│       ├── brightness_over_time.png
│       ├── contrast_over_time.png
│       └── streak_counts.png
└── night_YYYY-MM-DD/        # Additional observation nights
    └── ...
```

## Usage

### Creating a New Night Directory

1. Create a directory named `night_YYYY-MM-DD` where YYYY-MM-DD is the observation date
2. Create a `frames/` subdirectory within it
3. Place your time-lapse `.jpg` images in the `frames/` directory
4. Run the analysis from the tool directory:
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
- Read input frames
- Generate output files (masks, metrics, events)
- Store visualization plots
- Share data between different analysis tools

Each analysis tool in the repository can reference data directories here using relative paths like `../data/night_YYYY-MM-DD/`.
