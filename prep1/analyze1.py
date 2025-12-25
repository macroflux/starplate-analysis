# analyze1.py
import cv2
import json
import csv
from pathlib import Path
import numpy as np
import argparse
import yaml
import sys

def load_config(config_path: Path = None) -> dict:
    """
    Load configuration from YAML file or use defaults.
    
    Args:
        config_path: Path to config.yaml file. If None, looks for config.yaml in current directory.
    
    Returns:
        Configuration dictionary with all parameters.
    """
    import copy
    
    def deep_merge(base: dict, update: dict) -> dict:
        """Recursively merge update dict into base dict."""
        # Deep copy base to avoid mutating the original
        result = copy.deepcopy(base)
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = deep_merge(result[key], value)
            else:
                # Overwrite or add new key
                result[key] = value
        return result
    
    default_config = {
        "masking": {
            "brightness_threshold": 10,
            "overlay_region": {
                "top": 0,
                "bottom": 140,
                "left": 0,
                "right": 450
            }
        },
        "analysis": {
            "gaussian_sigma": 5
        },
        "streak_detection": {
            "canny_low": 40,
            "canny_high": 120,
            "hough_threshold": 60,
            "min_line_length": 40,
            "max_line_gap": 10
        },
        "events": {
            "min_streaks": 2
        }
    }
    
    if config_path is None:
        config_path = Path("config.yaml")
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                # Perform recursive deep merge
                if loaded_config:
                    default_config = deep_merge(default_config, loaded_config)
                print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default configuration.")
    else:
        print(f"Config file {config_path} not found. Using default configuration.")
    
    return default_config

def build_sky_mask(sample_img_path: Path, config: dict) -> np.ndarray:
    """
    One-time mask builder idea:
    - exclude very dark pixels (housing)
    - optionally exclude top-left overlay region
    Save and reuse this mask each run.
    """
    img = cv2.imread(str(sample_img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # crude sky vs housing mask
    threshold = config["masking"]["brightness_threshold"]
    mask = (gray > threshold).astype(np.uint8) * 255

    # remove overlay text region (tune these coords to your layout)
    overlay = config["masking"]["overlay_region"]
    mask[overlay["top"]:overlay["bottom"], overlay["left"]:overlay["right"]] = 0

    return mask

def star_contrast_score(gray: np.ndarray, mask: np.ndarray, config: dict) -> float:
    # high-pass: stars increase local contrast
    sigma = config["analysis"]["gaussian_sigma"]
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=sigma)
    hp = cv2.subtract(gray, blur)
    vals = hp[mask > 0]
    return float(np.std(vals))

def detect_streaks(gray: np.ndarray, mask: np.ndarray, config: dict):
    # super basic streak detection (starter):
    # edges -> Hough lines, return list of segments
    g = cv2.bitwise_and(gray, gray, mask=mask)
    
    params = config["streak_detection"]
    edges = cv2.Canny(g, params["canny_low"], params["canny_high"])
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                            threshold=params["hough_threshold"],
                            minLineLength=params["min_line_length"], 
                            maxLineGap=params["max_line_gap"])
    if lines is None:
        return []
    return [l[0].tolist() for l in lines]  # [x1,y1,x2,y2]

def main(night_dir: str, config_path: str = None):
    """
    Analyze astronomical time-lapse frames for transient events.
    
    Args:
        night_dir: Path to directory containing frames/ subdirectory
        config_path: Optional path to configuration YAML file
    """
    # Validate input directory first
    night = Path(night_dir)
    if not night.exists():
        print(f"Error: Directory '{night_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    frames_dir = night / "frames"
    if not frames_dir.exists():
        print(f"Error: Frames directory '{frames_dir}' does not exist.", file=sys.stderr)
        print(f"Expected structure: {night_dir}/frames/*.jpg", file=sys.stderr)
        sys.exit(1)
    
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        print(f"Error: No .jpg frames found in '{frames_dir}'.", file=sys.stderr)
        print("Make sure your image files are in JPEG format with .jpg extension.", file=sys.stderr)
        sys.exit(1)
    
    # Load configuration after validation
    config = load_config(Path(config_path) if config_path else None)
    
    print(f"Found {len(frames)} frames to process.")

    # Build or load mask
    mask_path = night / "sky_mask.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        print(f"Loaded existing sky mask from {mask_path}")
    else:
        print(f"Building sky mask from first frame...")
        mask = build_sky_mask(frames[0], config)
        cv2.imwrite(str(mask_path), mask)
        print(f"Saved sky mask to {mask_path}")

    metrics_rows = []
    events = []

    # Process frames with progress indicator
    print("\nProcessing frames:")
    for i, f in enumerate(frames, 1):
        if i % 10 == 0 or i == len(frames):
            print(f"  Progress: {i}/{len(frames)} ({100*i//len(frames)}%)")
        
        img = cv2.imread(str(f))
        if img is None:
            print(f"Warning: Could not read frame {f.name}, skipping.", file=sys.stderr)
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mean = float(np.mean(gray[mask > 0]))
        contrast = star_contrast_score(gray, mask, config)
        streaks = detect_streaks(gray, mask, config)

        metrics_rows.append({
            "file": f.name,
            "mean_brightness": mean,
            "star_contrast": contrast,
            "streak_count": len(streaks),
        })

        # Event detection based on configuration
        min_streaks = config["events"]["min_streaks"]
        if len(streaks) >= min_streaks:
            events.append({
                "file": f.name,
                "reason": "many_streaks",
                "streaks": streaks[:10],
            })

    # Handle edge case: no valid frames processed
    if not metrics_rows:
        print("Error: No frames were successfully processed.", file=sys.stderr)
        sys.exit(1)

    # Write outputs
    out_csv = night / "metrics.csv"
    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=metrics_rows[0].keys())
        w.writeheader()
        w.writerows(metrics_rows)

    out_events = night / "events.json"
    out_events.write_text(json.dumps(events, indent=2))

    print(f"\nAnalysis complete!")
    print(f"  Metrics saved to: {out_csv}")
    print(f"  Events saved to: {out_events}")
    print(f"  Detected {len(events)} events with {min_streaks}+ streaks")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze astronomical time-lapse frames for transient events (satellites, meteors, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze1.py ./night_2025-12-24
  python analyze1.py ./night_2025-12-24 --config custom_config.yaml
  
Output files:
  - sky_mask.png: Mask excluding non-sky regions
  - metrics.csv: Per-frame brightness, contrast, and streak statistics
  - events.json: Detected transient events with streak coordinates
"""
    )
    parser.add_argument(
        "night_dir",
        help="Path to night directory containing frames/ subdirectory with .jpg images"
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to configuration YAML file (default: ./config.yaml)",
        default=None
    )
    
    args = parser.parse_args()
    main(args.night_dir, args.config)
