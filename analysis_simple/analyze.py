# analyze.py
import cv2
import json
import csv
from pathlib import Path
from typing import Optional, List
import numpy as np
import argparse
import yaml
import sys
import importlib.util
from itertools import chain

# Import timelapse module for video generation
try:
    from tools.timelapse import build_timelapse
except ImportError:
    # Fallback for different directory structures
    timelapse_path = Path(__file__).parent / "tools" / "timelapse.py"
    if timelapse_path.exists():
        spec = importlib.util.spec_from_file_location("timelapse", timelapse_path)
        if spec and spec.loader:
            timelapse = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(timelapse)
            build_timelapse = timelapse.build_timelapse
    else:
        build_timelapse = None

def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file. If None, looks for config.yaml in current directory.
    
    Returns:
        Configuration dictionary with all parameters.
    
    Raises:
        FileNotFoundError: If config file doesn't exist
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
    
    if config_path is None:
        config_path = Path("config.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create a config.yaml file with required parameters."
        )
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if not config:
                raise ValueError(f"Config file {config_path} is empty")
            print(f"Loaded configuration from {config_path}")
            return config
    except Exception as e:
        raise RuntimeError(f"Error loading config from {config_path}: {e}")

def fill_holes(binary_255: np.ndarray) -> np.ndarray:
    """
    Fill holes in a binary 0/255 mask using flood fill from the border.
    
    Args:
        binary_255: Binary mask with values 0 or 255
    
    Returns:
        Binary mask with holes filled
    """
    mask = binary_255.copy()
    h, w = mask.shape

    # Floodfill needs a mask 2px larger than the image
    ff = np.zeros((h + 2, w + 2), np.uint8)

    # Flood-fill the background from (0,0)
    flood = mask.copy()
    cv2.floodFill(flood, ff, seedPoint=(0, 0), newVal=255)

    # Invert floodfilled image: holes become white
    flood_inv = cv2.bitwise_not(flood)

    # Combine original with holes
    filled = cv2.bitwise_or(mask, flood_inv)
    return filled

def remove_small_components(binary_255: np.ndarray, min_area: int = 800) -> np.ndarray:
    """
    Remove small connected components from a binary 0/255 image.

    This is used to drop tiny edge specks (often stars/noise) before we
    fill holes / thicken structures.

    Args:
        binary_255: Binary mask with values 0 or 255
        min_area: Minimum pixel area for a component to be kept

    Returns:
        Binary mask with small components removed
    """
    if min_area <= 0:
        return binary_255

    # Ensure we are working with 0/255 uint8
    img = (binary_255 > 0).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)

    out = np.zeros_like(img, dtype=np.uint8)
    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255

    return out

def build_sky_mask(frames: List[Path], config: dict) -> np.ndarray:
    """
    Build a robust sky mask using multiple frames.

    Uses a low-percentile image across the night to identify persistent
    dark obstructions (trees, housing, blockers) and removes them.
    """
    mcfg = config.get("masking", {})
    overlay = mcfg.get("overlay_region", {"top": 0, "bottom": 140, "left": 0, "right": 450})

    # --- sampling ---
    sample_count = int(mcfg.get("sample_count", 60))
    low_percentile = float(mcfg.get("low_percentile", 20))
    obstruction_thresh = int(mcfg.get("obstruction_threshold", 35))
    dilate_px = int(mcfg.get("dilate_px", 8))
    resize_width = int(mcfg.get("resize_width", 0))  # 0 = no resize

    paths = list(frames)
    if len(paths) == 0:
        raise ValueError("No frames provided to build_sky_mask()")

    if len(paths) > sample_count:
        idxs = np.linspace(0, len(paths) - 1, sample_count).astype(int)
        sample = [paths[i] for i in idxs]
    else:
        sample = paths

    grays = []
    target_shape = None
    
    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if resize_width and img.shape[1] > resize_width:
            scale = resize_width / img.shape[1]
            img = cv2.resize(img, (resize_width, int(img.shape[0] * scale)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Establish target shape from first valid frame
        if target_shape is None:
            target_shape = gray.shape
        
        # Skip frames with different dimensions
        if gray.shape != target_shape:
            print(f"Warning: Frame {p.name} has different dimensions {gray.shape} vs expected {target_shape}, skipping from mask generation.")
            continue
            
        grays.append(gray)

    if len(grays) < 5:
        raise ValueError("Not enough readable sample frames with consistent dimensions to build a mask.")

    stack = np.stack(grays, axis=0)  # (N,H,W)

    # --- persistent obstruction via low percentile (dark obstructions) ---
    low = np.percentile(stack, low_percentile, axis=0).astype(np.uint8)
    dark_obs = (low < obstruction_thresh)

    # --- persistent bright areas (lit trees, foreground) ---
    high_percentile = float(mcfg.get("high_percentile", 95))
    bright_thresh = int(mcfg.get("bright_obstruction_threshold", 235))
    high = np.percentile(stack, high_percentile, axis=0).astype(np.uint8)
    bright_obs = (high > bright_thresh)

    # Combine dark and bright obstructions
    obstruction = (dark_obs | bright_obs).astype(np.uint8) * 255
    sky = cv2.bitwise_not(obstruction)

    # --- remove overlay region ---
    sky[overlay["top"]:overlay["bottom"], overlay["left"]:overlay["right"]] = 0

    # --- morphology cleanup ---
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k, iterations=1)
    sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN,  k, iterations=1)

    # --- safety dilation of excluded regions (reduces tree edge leak) ---
    if dilate_px > 0:
        inv = cv2.bitwise_not(sky)
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
        inv = cv2.dilate(inv, kd, iterations=1)
        sky = cv2.bitwise_not(inv)

    return sky

def build_persistent_edge_mask(frames: List[Path], sky_mask: np.ndarray, config: dict) -> np.ndarray:
    """
    Build a mask of edges that appear persistently across multiple frames.
    
    This identifies static structures (trees, buildings, wires) that create
    repeated edges in the same locations, allowing us to exclude them from
    streak detection since real sky events don't repeat in the same pixels.
    
    Args:
        frames: List of frame paths to sample
        sky_mask: Base sky mask to constrain edge detection
        config: Configuration dictionary
    
    Returns:
        Binary mask where white pixels indicate persistent edges to exclude
    """
    pcfg = config.get("persistent_edges", {})
    sample_count = int(pcfg.get("sample_count", 80))
    canny_low = int(pcfg.get("canny_low", 40))
    canny_high = int(pcfg.get("canny_high", 120))
    hp_sigma = float(pcfg.get("hp_sigma", 2.0))
    keep_fraction = float(pcfg.get("keep_fraction", 0.20))  # edge must appear in >=20% samples
    dilate_px = int(pcfg.get("dilate_px", 6))

    paths = list(frames)
    if len(paths) == 0:
        raise ValueError("No frames provided for persistent edge mask.")

    if len(paths) > sample_count:
        idxs = np.linspace(0, len(paths) - 1, sample_count).astype(int)
        sample = [paths[i] for i in idxs]
    else:
        sample = paths

    edge_sum = np.zeros_like(sky_mask, dtype=np.uint16)
    used = 0

    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.shape != sky_mask.shape:
            continue

        g = cv2.bitwise_and(gray, gray, mask=sky_mask)

        # high-pass reduces slow gradients (clouds/glow)
        blur = cv2.GaussianBlur(g, (0, 0), sigmaX=hp_sigma)
        hp = cv2.subtract(g, blur)

        edges = cv2.Canny(hp, canny_low, canny_high)
        edge_sum += (edges > 0).astype(np.uint16)
        used += 1

    if used < 5:
        return np.zeros_like(sky_mask, dtype=np.uint8)

    thresh = max(1, int(np.ceil(keep_fraction * used)))
    persistent = (edge_sum >= thresh).astype(np.uint8) * 255

    # 1) Dilate to thicken the edge map
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
        persistent = cv2.dilate(persistent, k, iterations=1)

    # 2) CLOSE to connect nearby edge fragments (trees benefit a lot here)
    close_px = int(pcfg.get("close_px", max(8, dilate_px)))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px*2+1, close_px*2+1))
    persistent = cv2.morphologyEx(persistent, cv2.MORPH_CLOSE, kc, iterations=1)

    # 3) Remove tiny specks (stars/noise) BEFORE we fill holes.
    #    This prevents "edge speckle" from turning into giant filled regions.
    min_component_area = int(pcfg.get("min_component_area", 800))
    persistent = remove_small_components(persistent, min_area=min_component_area)

    # 4) Fill enclosed regions (turn outlines into solid blocks)
    persistent = fill_holes(persistent)

    # Constrain to sky_mask region
    persistent = cv2.bitwise_and(persistent, persistent, mask=sky_mask)

    return persistent

def star_contrast_score(gray: np.ndarray, mask: np.ndarray, config: dict) -> float:
    # high-pass: stars increase local contrast
    sigma = config["analysis"]["gaussian_sigma"]
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=sigma)
    hp = cv2.subtract(gray, blur)
    vals = hp[mask > 0]  # type: ignore
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

def run_tool_script(tool_path: Path, night_dir: Path, **kwargs):
    """
    Dynamically import and run a tool script.
    
    Args:
        tool_path: Path to the tool script
        night_dir: Path to night directory
        **kwargs: Additional arguments to pass to the tool
    """
    try:
        spec = importlib.util.spec_from_file_location(tool_path.stem, tool_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Most tools expect command-line args, so we'll call them as subprocesses instead
            return True
    except Exception as e:
        print(f"Warning: Could not load tool {tool_path.name}: {e}", file=sys.stderr)
        return False

def build_timelapse_video(
    frames_dir: Path,
    out_mp4: Path,
    fps: int = 30,
    pattern: str = "*.jpg",
    quality: int = 8
) -> bool:
    """
    Build an MP4 timelapse from image frames using OpenCV VideoWriter.
    
    This is a wrapper around tools.timelapse.build_timelapse for backward compatibility.
    
    Args:
        frames_dir: Directory containing image frames
        out_mp4: Output MP4 path
        fps: Frames per second
        pattern: Glob pattern for frames (default *.jpg)
        quality: Video quality parameter (currently not applied to encoding; reserved for future CRF mapping)
        
    Returns:
        True if successful, False otherwise
    """
    if build_timelapse is None:
        print("Error: Could not import timelapse module", file=sys.stderr)
        return False
    
    return build_timelapse(
        frames_dir=frames_dir,
        output_path=out_mp4,
        fps=fps,
        pattern=pattern,
        quality=quality,
        verbose=True
    )

def main(night_dir: str, config_path: Optional[str] = None, 
         validate: bool = False, visualize: bool = False, overlay: bool = False, timelapse: bool = False):
    """
    Analyze astronomical time-lapse frames for transient events.
    
    Args:
        night_dir: Path to directory containing frames/ subdirectory
        config_path: Optional path to configuration YAML file
        validate: Run validation checks before analysis
        visualize: Generate plots after analysis
        overlay: Create annotated images with detected streaks after analysis
        timelapse: Generate MP4 timelapse video after analysis
    """
    # Run validation if requested
    if validate:
        import subprocess
        tools_dir = Path(__file__).parent / "tools"
        validate_script = tools_dir / "validate_data.py"
        if validate_script.exists():
            print("\n" + "="*60)
            print("Running pre-analysis validation...")
            print("="*60)
            result = subprocess.run([sys.executable, str(validate_script), night_dir])
            if result.returncode != 0:
                print("\nValidation found issues. Fix them before proceeding.")
                sys.exit(1)
            print("\n" + "="*60)
            print("Validation passed! Proceeding with analysis...")
            print("="*60 + "\n")
    
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
    
    # Filter out frames with outlier dimensions (e.g., thumbnails)
    print(f"Found {len(frames)} frames. Checking dimensions...")
    frame_dims = {}
    frame_shapes = {}
    failed_to_load = []
    # Read each frame once to determine its dimensions and count occurrences
    for f in frames:
        img = cv2.imread(str(f))
        if img is not None:
            shape = (img.shape[0], img.shape[1])
            frame_shapes[f] = shape
            frame_dims[shape] = frame_dims.get(shape, 0) + 1
        else:
            failed_to_load.append(f.name)

    # Report frames that failed to load
    if failed_to_load:
        print(f"Warning: {len(failed_to_load)} frame(s) failed to load:")
        for f in failed_to_load[:5]:  # Show first 5
            print(f"  - {f}")
        if len(failed_to_load) > 5:
            print(f"  ... and {len(failed_to_load) - 5} more")

    if frame_dims:
        # Use the most common dimension
        common_dim = max(frame_dims.items(), key=lambda x: x[1])[0]
        print(f"Most common dimensions: {common_dim[1]}x{common_dim[0]} pixels")

        # Filter frames to only include those with common dimensions
        filtered_frames = []
        outliers = []
        for f in frames:
            shape = frame_shapes.get(f)
            # Skip frames that failed to load (already reported above)
            if shape is None:
                continue
            if shape == common_dim:
                filtered_frames.append(f)
            else:
                outliers.append(f.name)

        frames = filtered_frames

        if outliers:
            print(f"Filtered out {len(outliers)} frame(s) with different dimensions (likely thumbnails):")
            for o in outliers[:5]:  # Show first 5
                print(f"  - {o}")
            if len(outliers) > 5:
                print(f"  ... and {len(outliers) - 5} more")
    else:
        # No frame dimensions could be determined from any frames.
        # This indicates that none of the frames could be read by cv2.imread,
        # even though the frame paths were found on disk.
        print(
            "Error: Unable to read any frames for dimension checking. "
            "Please verify that the frame files are valid images and accessible.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not frames:
        print(f"Error: No valid frames remaining after filtering.", file=sys.stderr)
        sys.exit(1)
    print(f"Processing {len(frames)} frames with consistent dimensions.")
    
    # Load configuration with proper priority
    # Base: script_dir/config.yaml (required)
    # Override: night_dir/config.yaml (if exists, merges on top)
    # Final override: explicit --config argument (if provided, replaces base)
    
    import copy
    
    def deep_merge(base: dict, update: dict) -> dict:
        """Recursively merge update dict into base dict."""
        result = copy.deepcopy(base)
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    script_dir = Path(__file__).parent
    base_config_path = script_dir / "config.yaml"
    
    if config_path:
        # Explicit --config overrides everything
        config = load_config(Path(config_path))
    else:
        # Load base config from script directory (required)
        config = load_config(base_config_path)
        
        # If night directory has a config, merge it on top of base
        night_config = night / "config.yaml"
        if night_config.exists():
            print(f"Merging night-specific configuration from {night_config}")
            night_settings = load_config(night_config)
            config = deep_merge(config, night_settings)
    
    # Frames are now validated and configuration is loaded; proceed to rebuild masks.

    # Remove old masks to force regeneration with current parameters
    mask_path = night / "sky_mask.png"
    persistent_path = night / "persistent_edges.png"
    combined_path = night / "combined_mask.png"
    
    if mask_path.exists():
        mask_path.unlink()
        print(f"Removed old sky_mask.png to regenerate with current configuration.")
    if persistent_path.exists():
        persistent_path.unlink()
        print(f"Removed old persistent_edges.png to regenerate with current configuration.")
    if combined_path.exists():
        combined_path.unlink()
        print(f"Removed old combined_mask.png to regenerate with current configuration.")

    # Build masks (always regenerate to use current config)
    print(f"Building sky mask from multiple frames...")
    mask = build_sky_mask(frames, config)
    cv2.imwrite(str(mask_path), mask)
    print(f"Saved sky mask to {mask_path}")

    print("Building persistent edge mask from sampled frames...")
    print("(This identifies static structures like trees, buildings, wires to exclude from detection)")
    persistent_edges = build_persistent_edge_mask(frames, mask, config)
    cv2.imwrite(str(persistent_path), persistent_edges)
    print(f"Saved persistent edge mask to {persistent_path}")

    # Combined mask: keep sky where NOT persistent edges
    combined_mask = cv2.bitwise_and(mask, cv2.bitwise_not(persistent_edges))
    cv2.imwrite(str(combined_path), combined_mask)
    print(f"Saved combined mask to {combined_path}")
    print(
        f"mask px={np.count_nonzero(mask)} "
        f"persistent px={np.count_nonzero(persistent_edges)} "
        f"combined px={np.count_nonzero(combined_mask)}"
    )

    metrics_rows = []
    events = []
    
    # Event detection threshold from configuration
    min_streaks = config["events"]["min_streaks"]

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
        
        # Check if frame dimensions match the mask
        if gray.shape != combined_mask.shape:
            print(f"Warning: Frame {f.name} has different dimensions {gray.shape} vs mask {combined_mask.shape}, skipping.", file=sys.stderr)
            continue

        mean = float(np.mean(gray[combined_mask > 0]))  # type: ignore
        contrast = star_contrast_score(gray, combined_mask, config)
        streaks = detect_streaks(gray, combined_mask, config)

        metrics_rows.append({
            "file": f.name,
            "mean_brightness": mean,
            "star_contrast": contrast,
            "streak_count": len(streaks),
        })

        # Event detection based on configuration
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
    
    # Run post-analysis tools if requested
    tools_dir = Path(__file__).parent / "tools"
    
    if visualize:
        visualize_script = tools_dir / "visualize.py"
        if visualize_script.exists():
            import subprocess
            print("\n" + "="*60)
            print("Generating visualization plots...")
            print("="*60)
            plots_dir = night / "plots"
            result = subprocess.run(
                [sys.executable, str(visualize_script), str(night), "--output", str(plots_dir)]
            )
            if result.returncode == 0:
                print(f"  Plots saved to: {plots_dir}")
        else:
            print(f"Warning: visualize.py not found at {visualize_script}", file=sys.stderr)
    
    if overlay and len(events) > 0:
        overlay_script = tools_dir / "overlay_streaks.py"
        if overlay_script.exists():
            import subprocess
            print("\n" + "="*60)
            print("Creating annotated frames with detected streaks...")
            print("="*60)
            annotated_dir = night / "annotated"
            result = subprocess.run(
                [sys.executable, str(overlay_script), str(night), "--output", str(annotated_dir)]
            )
            if result.returncode == 0:
                print(f"  Annotated frames saved to: {annotated_dir}")
        else:
            print(f"Warning: overlay_streaks.py not found at {overlay_script}", file=sys.stderr)
    elif overlay and len(events) == 0:
        print("\nNote: No events detected, skipping overlay generation.")
    
    if timelapse:
        print("\n" + "="*60)
        print("Generating timelapse videos...")
        print("="*60)
        
        # Get timelapse settings from config
        timelapse_config = config.get("timelapse", {})
        fps = int(timelapse_config.get("fps", 30))
        quality = int(timelapse_config.get("quality", 8))
        
        # Generate raw frames timelapse
        raw_mp4 = night / "timelapse_raw.mp4"
        if build_timelapse_video(frames_dir, raw_mp4, fps=fps, quality=quality):
            print(f"  Raw timelapse saved to: {raw_mp4}")
        
        # Generate annotated frames timelapse if overlay was created
        annotated_dir = night / "annotated"
        # Ensure the annotated directory actually contains image frames before building timelapse
        has_annotated_frames = False
        if annotated_dir.exists():
            # Use chain to short-circuit on first match across multiple patterns
            has_annotated_frames = any(chain(
                annotated_dir.glob("*.png"),
                annotated_dir.glob("*.jpg"),
                annotated_dir.glob("*.jpeg")
            ))
        if overlay and len(events) > 0 and has_annotated_frames:
            annotated_mp4 = night / "timelapse_annotated.mp4"
            if build_timelapse_video(annotated_dir, annotated_mp4, fps=fps, quality=quality):
                print(f"  Annotated timelapse saved to: {annotated_mp4}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze astronomical time-lapse frames for transient events (satellites, meteors, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py ./night_2025-12-24
  python analyze.py ./night_2025-12-24 --config custom_config.yaml
  python analyze.py ./night_2025-12-24 --validate --visualize --overlay
  python analyze.py ./night_2025-12-24 --all-tools
  
Output files:
  - sky_mask.png: Mask excluding non-sky regions
  - metrics.csv: Per-frame brightness, contrast, and streak statistics
  - events.json: Detected transient events with streak coordinates
  - plots/ (with --visualize): Time-series plots of metrics
  - annotated/ (with --overlay): Frames with detected streaks drawn
"""
    )
    parser.add_argument(
        "night_dir",
        help="Path to night directory containing frames/ subdirectory with .jpg images"
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to configuration YAML file (default: config.yaml)",
        default=None
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks before analysis (uses tools/validate_data.py)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate time-series plots after analysis (uses tools/visualize.py)"
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Create annotated frames with detected streaks (uses tools/overlay_streaks.py)"
    )
    parser.add_argument(
        "--all-tools",
        action="store_true",
        help='Run all optional tools (validation, visualization, overlay, and timelapse)'
    )
    parser.add_argument(
        '--timelapse',
        action='store_true',
        help='Generate MP4 timelapse videos from frames and annotated images'
    )
    
    args = parser.parse_args()
    
    # --all-tools flag enables all optional tools
    if args.all_tools:
        args.validate = True
        args.visualize = True
        args.overlay = True
        args.timelapse = True
    
    main(args.night_dir, args.config, args.validate, args.visualize, args.overlay, args.timelapse)
