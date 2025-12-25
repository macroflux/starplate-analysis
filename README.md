# starplate-analysis

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

1. **Download Images** (if you don't have them yet):
   ```bash
   cd tools/data_fetch
   pip install -r requirements.txt
   python fetch.py 20251224
   ```

2. **Run Analysis**:
   ```bash
   cd ../../analysis_simple
   pip install -r requirements.txt
   python analyze.py ../data/night_2025-12-24/
   ```

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

## Data

The `data/` directory contains sample data structures and outputs shared across all tools. This is the common location where all analysis tools can generate and store their output images and results.

## Contributing

This is an experimental repository. Each tool may be at different stages of development. Check individual tool READMEs for status and contribution guidelines.

## License

See repository for license information.
