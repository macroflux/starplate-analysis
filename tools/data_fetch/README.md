# data_fetch

Robust data fetching tool for downloading astronomical time-lapse images from allsky.local server.

## Overview

This tool provides a reliable, configurable way to download astroplate images from a local allsky camera server. It includes retry logic, progress tracking, error handling, and various operational modes.

## Features

- **Configurable**: All settings in `config.yaml` with environment variable support
- **Robust Error Handling**: Exponential backoff, connection timeouts, network interruption handling
- **Progress Tracking**: Real-time progress bars using tqdm
- **Retry Logic**: Automatic tracking and retry of failed downloads
- **Dry Run Mode**: Preview what would be downloaded without actual downloads
- **Date Range Support**: Download multiple nights in a single command
- **Verbosity Levels**: Quiet, normal, and verbose output modes
- **Checksum Verification**: Optional MD5 checksum verification (configurable)
- **Summary Statistics**: Total size, duration, and download speed reporting
- **Module or Standalone**: Use as a command-line tool or import as a Python module

## Installation

### Prerequisites

- Python 3.7 or higher
- Network access to allsky.local server
- pip package manager

### Install Dependencies

```bash
cd tools/data_fetch
pip install -r requirements.txt
```

This installs:
- `requests>=2.31.0` - HTTP library for downloads
- `beautifulsoup4>=4.12.0` - HTML parsing for directory listings
- `pyyaml>=6.0` - Configuration file parsing
- `tqdm>=4.65.0` - Progress bars

## Usage

**Important:** The tool can be run from anywhere, but works best from the `tools/data_fetch/` directory.

### Basic Usage

Download a single night's images:

```bash
cd tools/data_fetch
python fetch.py 20251224
```

### Verbose Output

```bash
python fetch.py 20251224 --verbose
```

### Quiet Mode

Minimal output, only errors:

```bash
python fetch.py 20251224 --quiet
```

### Retry Failed Downloads

If a previous download had failures, retry them:

```bash
python fetch.py 20251224 --retry
```

### Dry Run Mode

Preview what would be downloaded without downloading:

```bash
python fetch.py 20251224 --dry-run
```

### Date Range Download

Download multiple nights:

```bash
python fetch.py 20251224 --end-date 20251226
```

This downloads images for:
- 2025-12-24
- 2025-12-25
- 2025-12-26

### Custom Configuration

Use a custom config file:

```bash
python fetch.py 20251224 --config my_config.yaml
```

### Force Re-download

Re-download even if files already exist:

```bash
python fetch.py 20251224 --force
```

### Combine Options

```bash
python fetch.py 20251224 --end-date 20251226 --verbose --force
```

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
# Base URL for image downloads
base_url: "http://allsky.local/images/"

# Network settings
network:
  timeout: 30  # seconds for image download
  listing_timeout: 10  # seconds for directory listing
  max_retries: 3
  backoff_multiplier: 2
  initial_backoff: 1

# Supported image file extensions
image_extensions:
  - .jpg
  - .jpeg
  - .png

# Download settings
download:
  verify_checksum: false
  chunk_size: 8192

# Output settings
output:
  data_dir: "../../data"
  night_dir_format: "night_{date}"
  frames_subdir: "frames"
  failed_log_filename: "failed_downloads.json"
```

### Environment Variable Overrides

Override configuration with environment variables:

```bash
export BASE_URL="http://custom-server.local/images/"
export NETWORK_TIMEOUT=60
python fetch.py 20251224
```

Supported environment variables:
- `BASE_URL` - Override base URL
- `NETWORK_TIMEOUT` - Override timeout
- `DATA_DIR` - Override data directory

## Output Structure

Downloaded images are organized as:

```
data/
└── night_2025-12-24/
    ├── frames/
    │   ├── image001.jpg
    │   ├── image002.jpg
    │   └── ...
    └── failed_downloads.json  (if any downloads failed)
```

### Failed Downloads Log

If any downloads fail, they're logged to `failed_downloads.json`:

```json
[
  {
    "filename": "image123.jpg",
    "url": "http://allsky.local/images/20251224/image123.jpg",
    "error": "Connection timeout"
  }
]
```

Use `--retry` to retry these failed downloads.

## Using as a Python Module

Import and use in your own scripts:

```python
from fetch import ImageFetcher

# Create fetcher instance
fetcher = ImageFetcher(
    config_path="config.yaml",
    verbose=True
)

# Download a single night
success = fetcher.download_night("20251224")

# Download a date range
fetcher.download_date_range("20251224", "20251226")

# Retry failed downloads
fetcher.retry_failed("20251224")
```

## Command-Line Reference

```
usage: fetch.py [-h] [--config CONFIG] [--verbose | --quiet] [--retry]
                [--dry-run] [--end-date END_DATE] [--force]
                date

Download astroplate images from allsky.local

positional arguments:
  date                 Start date in YYYYMMDD format (e.g., 20251224)

optional arguments:
  -h, --help           show this help message and exit
  --config CONFIG      Path to config file (default: config.yaml)
  --verbose, -v        Verbose output
  --quiet, -q          Quiet output (errors only)
  --retry, -r          Retry previously failed downloads
  --dry-run, -n        Preview downloads without downloading
  --end-date END_DATE  End date for range download (YYYYMMDD)
  --force, -f          Force re-download existing files
```

## Troubleshooting

### Connection Refused / Timeout

- Check that allsky.local is accessible: `ping allsky.local`
- Verify the base URL in config.yaml
- Try increasing `network.timeout` in config.yaml

### No Images Found

- Check the date format (must be YYYYMMDD)
- Verify images exist on the server for that date
- Use `--verbose` to see the directory listing response

### Permission Denied

- Ensure you have write permissions to the data directory
- Check that the data directory path in config.yaml is correct

### Failed Downloads Persist

- Use `--retry` to retry failed downloads
- Check `failed_downloads.json` for error details
- Try with `--verbose` to see detailed error messages

## Integration with Analysis Tools

This tool integrates with the analysis pipeline:

```bash
# 1. Download images
cd tools/data_fetch
python fetch.py 20251224

# 2. Run analysis
cd ../../analysis_simple
python analyze.py ../data/night_2025-12-24/
```

## Examples

### Download Last Week

```bash
# Download 7 days of data
python fetch.py 20251218 --end-date 20251224 --verbose
```

### Quiet Batch Processing

```bash
# For cron jobs or scripts
python fetch.py 20251224 --quiet
if [ $? -eq 0 ]; then
    echo "Download successful"
else
    echo "Download failed"
fi
```

### Development/Testing

```bash
# See what would be downloaded
python fetch.py 20251224 --dry-run --verbose
```

## Contributing

This tool follows the repository's tool development guidelines. See the main README for details.

## License

See repository for license information.
