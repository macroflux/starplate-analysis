# starplate-analysis

A collection of experimental tools for astronomical plate/time-lapse analysis to detect satellites, meteors, and other transient events.

## Repository Structure

This repository contains multiple analysis tools, each in its own self-contained folder with its own dependencies, configuration, and documentation.

### Available Tools

#### `analysis_simple/`
Basic computer vision pipeline using OpenCV for streak detection in time-lapse astronomical images.

- **Purpose**: Detect satellite streaks, meteors, and transient events using edge detection and Hough transforms
- **Dependencies**: OpenCV, NumPy, Matplotlib, PyYAML
- **Status**: Active development (v1)

See [`analysis_simple/README.md`](analysis_simple/README.md) for detailed usage and configuration.

#### Future Tools (Planned)
- `analysis_ml/` - Machine learning-based detection using neural networks
- `analysis_sk/` - scikit-learn based statistical analysis
- Additional experimental pipelines as they're developed

## Getting Started

Each tool is independent and should be run from within its own directory:

```bash
# Navigate into the tool directory
cd analysis_simple/

# Install dependencies
pip install -r requirements.txt

# Run the tool
python analyze.py ../data/night_2025-12-24/
```

## Tool Development Guidelines

Each tool folder should contain:
- `README.md` - Complete usage documentation
- `requirements.txt` - Python dependencies
- `config.yaml` or similar - Configuration files
- `tools/` - Tool-specific utilities and helper scripts
- `.github/` (optional) - Tool-specific CI/CD workflows
- Source code files

Tools should be self-contained and independently runnable.

## Data

The `data/` directory contains sample data structures and outputs shared across all tools. This is the common location where all analysis tools can generate and store their output images and results.

## Contributing

This is an experimental repository. Each tool may be at different stages of development. Check individual tool READMEs for status and contribution guidelines.

## License

See repository for license information.
