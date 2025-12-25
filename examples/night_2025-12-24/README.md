# Example Night Data Structure

This directory demonstrates the expected structure for a night's worth of time-lapse frames.

## Directory Structure

```
night_2025-12-24/
├── frames/              # INPUT: Place your .jpg frames here
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── ...
├── sky_mask.png         # OUTPUT: Generated mask (example provided)
├── metrics.csv          # OUTPUT: Per-frame statistics (example provided)
└── events.json          # OUTPUT: Detected events (example provided)
```

## Usage

### For Real Data

1. Create a directory for your observation night
2. Create a `frames/` subdirectory
3. Place your `.jpg` image files in `frames/`
4. Run the analysis:
   ```bash
   python prep1/analyze1.py ./night_2025-12-24
   ```

### For Testing

This example directory includes sample output files (`metrics.csv`, `events.json`, `sky_mask.png`) to show what the analysis produces. These are synthetic examples for reference.

## Image Requirements

- **Format**: JPEG with `.jpg` extension
- **Naming**: Any naming scheme works (files are sorted alphabetically)
- **Content**: Night sky images from a fixed camera position
- **Sequence**: Should be a time-series (sequential frames)

## Expected Outputs

After running analysis, you'll get:

1. **sky_mask.png**: Binary mask showing which pixels are analyzed
2. **metrics.csv**: Statistics for each frame (brightness, contrast, streak count)
3. **events.json**: List of frames with detected transient events

## Tips

- Use consistent exposure settings across frames
- Ensure camera is stationary (mount on tripod)
- Higher resolution images provide better detection
- Process one night at a time for manageable file sizes
