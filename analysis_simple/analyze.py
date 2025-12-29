"""
Astronomical Time-Lapse Analysis Pipeline with Activity Detection

This module provides a comprehensive computer vision pipeline for analyzing astronomical
time-lapse images to detect and catalog transient events such as satellites, meteors,
and aircraft. The pipeline includes automatic activity window detection using interest
scoring to identify high-value observation periods.

Workflow Overview
-----------------
1. **Mask Building**
   - Sky mask: Exclude camera housing, overlays, and dark obstructions
   - Persistent edge mask: Exclude static structures (trees, buildings, wires)
   - Combined mask: Final analysis region

2. **Frame Analysis**
   - Compute per-frame metrics: brightness, contrast, focus, streak count
   - Detect linear streaks using Canny edge detection and Hough transforms
   - Apply combined mask to exclude false positives from static structures

3. **Activity Detection**
   - Compute interest scores: weighted combination of metrics and changes
   - Detect activity windows: contiguous periods above interest threshold
   - Generate per-window artifacts: timelapses, keograms, startrails

4. **Output Generation**
   - Structured data files: metrics.csv, events.json, activity_windows.json
   - Analysis masks: sky_mask.png, persistent_edges.png, combined_mask.png
   - Per-window artifacts: timelapse, keogram, startrail for each activity period
   - Optional: time-series plots, annotated frames, full-night timelapses

Output Structure
----------------
night_YYYY-MM-DD/
├── frames/              # INPUT: Raw image sequence
├── masks/               # Analysis masks (sky, persistent edges, combined)
├── data/                # Structured data outputs (metrics, events, windows)
├── activity/            # Per-window artifacts for high-interest periods
├── plots/               # Time-series visualizations (optional)
├── annotated/           # Frames with detected streaks overlaid (optional)
└── timelapse/           # Full-night timelapse videos (optional)

Machine Learning Preparation
----------------------------
The structured outputs are designed for ML workflows:

- **Training Data Extraction**: Use activity_windows.json to identify high-value frames
- **Feature Engineering**: metrics.csv provides per-frame features with interest scores
- **Event Classification**: events.json contains labeled transient events with coordinates
- **Automated Annotation**: Per-window keograms and startrails aid visual validation
- **Balanced Sampling**: Easy positive/negative split using activity windows

Configuration
-------------
All parameters are configurable via config.yaml:

- masking: Sky mask and persistent edge detection parameters
- streak_detection: Canny and Hough transform thresholds
- windows: Activity detection thresholds and weights
- keogram: Center column sampling width
- startrails: Gamma correction for visibility enhancement
- timelapse: Video FPS and quality settings

See README.md for comprehensive configuration documentation and tuning guidance.

Usage
-----
Basic analysis:
    python analyze.py night_2025-12-24/

With all optional tools:
    python analyze.py night_2025-12-24/ --all-tools

Custom configuration:
    python analyze.py night_2025-12-24/ --config custom_config.yaml

For detailed usage, configuration, and ML integration examples, see:
    analysis_simple/README.md
"""

import cv2
import json
import csv
from pathlib import Path
from typing import Optional, List, Dict, Tuple
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

# --- NEW METRICS FUNCTIONS ---

def focus_score(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Focus proxy: variance of Laplacian in masked region.
    Higher = sharper stars; lower = blur/cloud/defocus.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    vals = lap[mask > 0]
    if vals.size == 0:
        return 0.0
    return float(vals.var())

def safe_mean(gray: np.ndarray, mask: np.ndarray) -> float:
    """Safe mean calculation with NaN handling."""
    vals = gray[mask > 0]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))

def robust_zscore(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Robust z-score using median/MAD (stable under outliers).
    """
    x = x.astype(np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad + eps  # MAD->sigma approx
    return (x - med) / scale

def diff1(x: np.ndarray) -> np.ndarray:
    """
    First difference with same-length output (prepend first value).
    """
    x = x.astype(np.float64)
    if x.size == 0:
        return x
    d = np.diff(x, prepend=x[0])
    return d

def detect_activity_windows(
    score: np.ndarray,
    threshold: float,
    min_len: int = 6,
    pad: int = 10,
    merge_gap: int = 8,
    max_windows: int = 10
):
    """
    Convert a 1D score signal into merged/padded windows where score >= threshold.
    Returns list of dict windows with start/end/peak.
    """
    n = len(score)
    if n == 0:
        return []

    active = score >= threshold
    idx = np.where(active)[0]
    if idx.size == 0:
        return []

    # Find contiguous runs
    runs = []
    start = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
        else:
            runs.append((start, prev))
            start = k
            prev = k
    runs.append((start, prev))

    # Filter by min_len and pad
    padded = []
    for s, e in runs:
        if (e - s + 1) < min_len:
            continue
        ps = max(0, s - pad)
        pe = min(n - 1, e + pad)
        padded.append((ps, pe))

    if not padded:
        return []

    # Merge nearby windows
    merged = [padded[0]]
    for s, e in padded[1:]:
        ms, me = merged[-1]
        if s <= me + merge_gap:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))

    # Convert to dicts with peaks
    windows = []
    for s, e in merged:
        seg = score[s:e+1]
        peak_rel = int(np.argmax(seg))
        peak_idx = s + peak_rel
        windows.append({
            "start": int(s),
            "end": int(e),
            "peak_index": int(peak_idx),
            "peak_value": float(score[peak_idx]),
            "length": int(e - s + 1),
        })

    # Sort by peak desc and cap
    windows.sort(key=lambda w: w["peak_value"], reverse=True)
    windows = windows[:max_windows]

    # Keep chronological order for reporting/output
    windows.sort(key=lambda w: w["start"])
    return windows

def load_interest_windows(windows_path: Path) -> List[dict]:
    """
    Load interest-based activity windows from activity_windows.json.
    
    Args:
        windows_path: Path to activity_windows.json
    
    Returns:
        List of window dictionaries
    """
    if not windows_path.exists():
        return []
    try:
        with open(windows_path, 'r') as f:
            windows = json.load(f)
        # Ensure each window has a 'source' field
        for w in windows:
            if 'source' not in w:
                w['source'] = 'interest'
        return windows
    except Exception as e:
        print(f"Warning: Failed to load interest windows from {windows_path}: {e}", file=sys.stderr)
        return []

def load_ml_windows(windows_path: Path) -> List[dict]:
    """
    Load ML-based activity windows from ml_windows.json.
    
    Args:
        windows_path: Path to ml_windows.json
    
    Returns:
        List of window dictionaries
    """
    if not windows_path.exists():
        return []
    try:
        with open(windows_path, 'r') as f:
            windows = json.load(f)
        # Ensure each window has a 'source' field
        for w in windows:
            if 'source' not in w:
                w['source'] = 'ml'
        return windows
    except Exception as e:
        print(f"Warning: Failed to load ML windows from {windows_path}: {e}", file=sys.stderr)
        return []

def calculate_iou(window1: dict, window2: dict) -> float:
    """
    Calculate Intersection over Union (IoU) for two windows.
    
    Windows use inclusive frame indices (both start and end are included).
    For example, window {'start': 0, 'end': 10} contains frames [0, 1, 2, ..., 10].
    
    Args:
        window1: First window dict with 'start' and 'end' keys (inclusive)
        window2: Second window dict with 'start' and 'end' keys (inclusive)
    
    Returns:
        IoU value between 0.0 and 1.0
    """
    s1, e1 = window1['start'], window1['end']
    s2, e2 = window2['start'], window2['end']
    
    # Calculate intersection
    intersection_start = max(s1, s2)
    intersection_end = min(e1, e2)
    
    if intersection_start > intersection_end:
        return 0.0  # No overlap
    
    intersection = intersection_end - intersection_start + 1  # +1 because indices are inclusive
    
    # Calculate union
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - intersection  # +1 because indices are inclusive
    
    return float(intersection) / float(union)

def merge_windows_simple(windows: List[Tuple[int, int]], merge_gap: int = 8) -> List[Tuple[int, int]]:
    """
    Merge overlapping or nearby windows.
    
    Args:
        windows: List of (start, end) tuples
        merge_gap: Merge windows within this distance
    
    Returns:
        List of merged (start, end) tuples
    """
    if not windows:
        return []
    
    windows = sorted(windows, key=lambda t: t[0])
    merged = [windows[0]]
    
    for s, e in windows[1:]:
        ms, me = merged[-1]
        if s <= me + merge_gap:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    
    return merged

def select_windows(
    source: str,
    interest_windows: List[dict],
    ml_windows: List[dict],
    iou_threshold: float = 0.3,
    merge_gap: int = 8
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Select windows based on the specified source strategy.
    
    Args:
        source: One of 'interest', 'ml', or 'hybrid'
        interest_windows: Interest-based windows
        ml_windows: ML-based windows
        iou_threshold: IoU threshold for hybrid mode (default 0.3)
        merge_gap: Gap for merging windows in hybrid mode
    
    Returns:
        Tuple of (selected_windows, stats_dict)
        where stats_dict contains counts of windows from each source
    """
    stats = {'interest': 0, 'ml': 0, 'hybrid_added': 0}
    
    if source == 'interest':
        stats['interest'] = len(interest_windows)
        return interest_windows, stats
    
    elif source == 'ml':
        stats['ml'] = len(ml_windows)
        return ml_windows, stats
    
    elif source == 'hybrid':
        # Start with ML windows (higher confidence typically)
        result = list(ml_windows)
        stats['ml'] = len(ml_windows)
        
        # Add interest windows that don't significantly overlap with ML windows
        for iw in interest_windows:
            # Calculate IoU with all ML windows
            max_iou = 0.0
            for mw in ml_windows:
                iou = calculate_iou(iw, mw)
                max_iou = max(max_iou, iou)
            
            # If IoU is below threshold, this is a novel window
            if max_iou < iou_threshold:
                result.append(iw)
                stats['hybrid_added'] += 1
        
        # Merge any overlapping windows
        if result:
            # Convert to (start, end) tuples for merging
            tuples = [(w['start'], w['end']) for w in result]
            merged_tuples = merge_windows_simple(tuples, merge_gap=merge_gap)
            
            # Convert back to window dicts, preserving source info where possible
            final_windows = []
            for s, e in merged_tuples:
                # Find the best matching original window to preserve metadata
                best_match = None
                best_overlap = 0
                for w in result:
                    overlap = max(0, min(w['end'], e) - max(w['start'], s) + 1)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = w
                
                if best_match:
                    # Use the best match as template, update start/end
                    merged_w = dict(best_match)
                    merged_w['start'] = int(s)
                    merged_w['end'] = int(e)
                    merged_w['length'] = int(e - s + 1)
                    merged_w['source'] = 'hybrid'
                    final_windows.append(merged_w)
                else:
                    # Shouldn't happen, but create a minimal window
                    final_windows.append({
                        'start': int(s),
                        'end': int(e),
                        'length': int(e - s + 1),
                        'source': 'hybrid'
                    })
            
            # Sort chronologically
            final_windows.sort(key=lambda w: w['start'])
            stats['interest'] = len(interest_windows)
            return final_windows, stats
        
        return result, stats
    
    else:
        raise ValueError(f"Invalid windows source: {source}. Must be 'interest', 'ml', or 'hybrid'")

def ensure_dir(p: Path) -> Path:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_gray_frame(path: Path, target_shape=None) -> Optional[np.ndarray]:
    """Load frame as grayscale with optional shape validation."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if target_shape is not None and g.shape != target_shape:
        return None
    return g

def build_keogram(frames: List[Path], mask: np.ndarray, out_png: Path, column_width: int = 3) -> bool:
    """
    Keogram: for each frame, take the vertical slice at center (avg over column_width)
    and stack slices horizontally (time axis).
    """
    h, w = mask.shape
    cx = w // 2
    half = max(1, column_width // 2)

    strips = []
    for p in frames:
        g = load_gray_frame(p, target_shape=(h, w))
        if g is None:
            continue
        # apply mask
        gm = cv2.bitwise_and(g, g, mask=mask)
        x0 = max(0, cx - half)
        x1 = min(w, cx + half + 1)
        strip = gm[:, x0:x1]
        # collapse slice to 1 column by averaging
        col = np.mean(strip.astype(np.float32), axis=1).astype(np.uint8)  # (H,)
        strips.append(col)

    if len(strips) < 2:
        return False

    keo = np.stack(strips, axis=1)  # (H, T)
    cv2.imwrite(str(out_png), keo)
    return True

def build_startrails(frames: List[Path], mask: np.ndarray, out_png: Path, gamma: float = 1.0) -> bool:
    """
    Startrails: per-pixel max over time (masked).
    Optionally apply a simple gamma.
    """
    h, w = mask.shape
    acc = np.zeros((h, w), dtype=np.uint8)

    used = 0
    for p in frames:
        g = load_gray_frame(p, target_shape=(h, w))
        if g is None:
            continue
        gm = cv2.bitwise_and(g, g, mask=mask)
        acc = np.maximum(acc, gm)
        used += 1

    if used < 2:
        return False

    if gamma and abs(gamma - 1.0) > 1e-3:
        f = acc.astype(np.float32) / 255.0
        f = np.power(f, 1.0 / gamma)
        acc = np.clip(f * 255.0, 0, 255).astype(np.uint8)

    cv2.imwrite(str(out_png), acc)
    return True

def build_timelapse_from_list(frame_paths: List[Path], out_mp4: Path, fps: int = 30) -> bool:
    """Build timelapse from a list of frame paths."""
    if not frame_paths:
        return False
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return False
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
    if not vw.isOpened():
        return False

    wrote = 0
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            continue
        vw.write(img)
        wrote += 1

    vw.release()
    return wrote > 0

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
         validate: bool = False, visualize: bool = False, overlay: bool = False, timelapse: bool = False,
         windows_source: str = 'interest'):
    """
    Analyze astronomical time-lapse frames for transient events.
    
    Args:
        night_dir: Path to directory containing frames/ subdirectory
        config_path: Optional path to configuration YAML file
        validate: Run validation checks before analysis
        visualize: Generate plots after analysis
        overlay: Create annotated images with detected streaks after analysis
        timelapse: Generate MP4 timelapse video after analysis
        windows_source: Window source ('interest', 'ml', or 'hybrid')
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

    # Create masks directory and remove old masks to force regeneration with current parameters
    masks_dir = ensure_dir(night / "masks")
    mask_path = masks_dir / "sky_mask.png"
    persistent_path = masks_dir / "persistent_edges.png"
    combined_path = masks_dir / "combined_mask.png"
    
    if mask_path.exists():
        mask_path.unlink()
        print(f"Removed old masks/sky_mask.png to regenerate with current configuration.")
    if persistent_path.exists():
        persistent_path.unlink()
        print(f"Removed old masks/persistent_edges.png to regenerate with current configuration.")
    if combined_path.exists():
        combined_path.unlink()
        print(f"Removed old masks/combined_mask.png to regenerate with current configuration.")

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

        # UPDATED: Use safe_mean and add focus_score
        mean = safe_mean(gray, combined_mask)
        contrast = star_contrast_score(gray, combined_mask, config)
        focus = focus_score(gray, combined_mask)
        streaks = detect_streaks(gray, combined_mask, config)

        metrics_rows.append({
            "file": f.name,
            "mean_brightness": mean,
            "star_contrast": contrast,
            "focus_score": focus,
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

    # --- STEP 2: Compute interest scores ---
    print("\nComputing activity interest scores...")
    files = [r["file"] for r in metrics_rows]
    mean_arr = np.array([r["mean_brightness"] for r in metrics_rows], dtype=np.float64)
    contr_arr = np.array([r["star_contrast"] for r in metrics_rows], dtype=np.float64)
    focus_arr = np.array([r["focus_score"] for r in metrics_rows], dtype=np.float64)
    streak_arr = np.array([r["streak_count"] for r in metrics_rows], dtype=np.float64)

    # Change signals (spikes are usually in deltas)
    d_mean = diff1(mean_arr)
    d_contr = diff1(contr_arr)
    d_focus = diff1(focus_arr)

    # Robust z-scores
    z_streak = robust_zscore(streak_arr)
    z_dmean  = robust_zscore(np.abs(d_mean))
    z_dcontr = robust_zscore(np.abs(d_contr))
    z_dfocus = robust_zscore(np.abs(d_focus))

    # Weights (configurable)
    wcfg = config.get("windows", {})
    w_streak = float(wcfg.get("w_streak", 0.60))
    w_mean   = float(wcfg.get("w_mean",   0.15))
    w_contr  = float(wcfg.get("w_contrast",0.15))
    w_focus  = float(wcfg.get("w_focus",  0.10))

    interest = (w_streak * z_streak) + (w_mean * z_dmean) + (w_contr * z_dcontr) + (w_focus * z_dfocus)

    # Attach to metrics rows (so it lands in CSV)
    for i in range(len(metrics_rows)):
        metrics_rows[i]["interest_score"] = float(interest[i])
        metrics_rows[i]["z_streak"] = float(z_streak[i])

    # --- STEP 3: Detect activity windows ---
    threshold = float(wcfg.get("threshold", 3.5))
    min_len = int(wcfg.get("min_len", 6))
    pad = int(wcfg.get("pad", 10))
    merge_gap = int(wcfg.get("merge_gap", 8))
    max_windows = int(wcfg.get("max_windows", 10))

    interest_windows = detect_activity_windows(
        interest,
        threshold=threshold,
        min_len=min_len,
        pad=pad,
        merge_gap=merge_gap,
        max_windows=max_windows
    )

    # Write outputs to data subfolder
    data_dir = ensure_dir(night / "data")
    out_csv = data_dir / "metrics.csv"
    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=metrics_rows[0].keys())
        w.writeheader()
        w.writerows(metrics_rows)

    out_events = data_dir / "events.json"
    out_events.write_text(json.dumps(events, indent=2))

    # Always save interest-based windows to activity_windows.json
    out_windows = data_dir / "activity_windows.json"
    out_windows.write_text(json.dumps(interest_windows, indent=2))

    # --- STEP 3b: Window Source Selection ---
    # Load ML windows if needed
    ml_windows = []
    if windows_source in ('ml', 'hybrid'):
        ml_windows_path = data_dir / "ml_windows.json"
        if not ml_windows_path.exists():
            if windows_source == 'ml':
                print(f"\nError: ML windows file not found: {ml_windows_path}", file=sys.stderr)
                print(f"Please run ML classifier first:", file=sys.stderr)
                print(f"  cd analysis_ml_activity_classifier && python train.py {night_dir}", file=sys.stderr)
                print(f"  cd ../analysis_ml_windows && python infer_windows.py {night_dir}", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"\nWarning: ML windows file not found: {ml_windows_path}", file=sys.stderr)
                print(f"Falling back to interest-only windows for hybrid mode.", file=sys.stderr)
                windows_source = 'interest'
        else:
            ml_windows = load_ml_windows(ml_windows_path)
    
    # Select windows based on source
    iou_threshold = float(wcfg.get("hybrid_iou_threshold", 0.3))
    windows, window_stats = select_windows(
        windows_source,
        interest_windows,
        ml_windows,
        iou_threshold=iou_threshold,
        merge_gap=merge_gap
    )
    
    # Save hybrid windows if applicable
    if windows_source == 'hybrid' and windows:
        hybrid_windows_path = data_dir / "windows_hybrid.json"
        hybrid_windows_path.write_text(json.dumps(windows, indent=2))
        print(f"\nHybrid windows saved to: {hybrid_windows_path}")
    
    # Display window source information
    print(f"\n{'='*60}")
    print(f"Window Source: {windows_source}")
    print(f"{'='*60}")
    
    if windows_source == 'interest':
        print(f"Loaded {window_stats['interest']} interest-based windows")
    elif windows_source == 'ml':
        print(f"Loaded {window_stats['ml']} ML-based windows")
    elif windows_source == 'hybrid':
        print(f"Loaded {window_stats['ml']} ML-based windows")
        print(f"Loaded {window_stats['interest']} interest-based windows")
        print(f"Added {window_stats['hybrid_added']} non-overlapping interest windows (IoU < {iou_threshold})")
        print(f"Final merged count: {len(windows)} windows")

    print(f"\nAnalysis complete!")
    print(f"  Data directory: {data_dir}")
    print(f"  Metrics saved to: {out_csv}")
    print(f"  Events saved to: {out_events}")
    print(f"  Activity windows saved to: {out_windows}")
    print(f"  Detected {len(events)} events with {min_streaks}+ streaks")

    # Display detected windows
    if windows:
        print("\n" + "="*60)
        print(f"Activity windows detected ({windows_source}):")
        print("="*60)
        for w in windows:
            source_label = f" [{w.get('source', 'unknown')}]" if 'source' in w else ""
            peak_str = f"peak@{w['peak_index']}={w['peak_value']:.2f}" if 'peak_value' in w else ""
            print(
                f"  frames {w['start']}–{w['end']} "
                f"(len={w['length']}){source_label} {peak_str}"
            )
    else:
        print("\nNo activity windows detected above threshold.")

    # Run post-analysis tools if requested
    tools_dir = Path(__file__).parent / "tools"

    # ------------------------------------------------------------
    # OVERLAY FIRST (so Step 4 can build annotated window clips)
    # ------------------------------------------------------------
    annotated_dir = night / "annotated"
    if overlay and len(events) > 0:
        overlay_script = tools_dir / "overlay_streaks.py"
        if overlay_script.exists():
            import subprocess
            print("\n" + "="*60)
            print("Creating annotated frames with detected streaks...")
            print("="*60)
            result = subprocess.run(
                [sys.executable, str(overlay_script), str(night), "--output", str(annotated_dir)]
            )
            if result.returncode == 0:
                print(f"  Annotated frames saved to: {annotated_dir}")
            else:
                print("  Warning: overlay tool returned non-zero exit code.", file=sys.stderr)
        else:
            print(f"Warning: overlay_streaks.py not found at {overlay_script}", file=sys.stderr)
    elif overlay and len(events) == 0:
        print("\nNote: No events detected, skipping overlay generation.")

    # ------------------------------------------------------------
    # VISUALIZE (optional)
    # ------------------------------------------------------------
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
                print("  Warning: visualize tool returned non-zero exit code.", file=sys.stderr)
        else:
            print(f"Warning: visualize.py not found at {visualize_script}", file=sys.stderr)

    # ------------------------------------------------------------
    # --- STEP 4: Generate per-window artifacts ---
    # (NOW runs after overlay, so annotated window clips can work)
    # ------------------------------------------------------------
    if windows:
        print("\n" + "="*60)
        print("Generating per-window artifacts...")
        print("="*60)

        activity_root = ensure_dir(night / "activity")

        timelapse_cfg = config.get("timelapse", {})
        fps = int(timelapse_cfg.get("fps", 30))

        column_width = int(config.get("keogram", {}).get("column_width", 3))
        trails_gamma = float(config.get("startrails", {}).get("gamma", 1.0))

        # Determine if annotated frames exist (post-overlay)
        has_annotated_frames = False
        if annotated_dir.exists():
            has_annotated_frames = any(chain(
                annotated_dir.glob("*.png"),
                annotated_dir.glob("*.jpg"),
                annotated_dir.glob("*.jpeg")
            ))

        for wi, w in enumerate(windows):
            s, e = w["start"], w["end"]
            window_frames = frames[s:e+1]

            win_dir = ensure_dir(activity_root / f"window_{wi:02d}_{s:04d}_{e:04d}")
            print(f"\n  Window {wi}: frames {s}–{e} ({len(window_frames)} frames)")

            # 1) subset timelapse (raw frames)
            out_mp4 = win_dir / "timelapse_window.mp4"
            ok_vid = build_timelapse_from_list(window_frames, out_mp4, fps=fps)
            if ok_vid:
                print(f"    ✓ timelapse: {out_mp4.name}")
            else:
                print(f"    ✗ timelapse failed: {out_mp4.name}", file=sys.stderr)

            # 2) keogram
            keo_png = win_dir / "keogram.png"
            ok_keo = build_keogram(window_frames, combined_mask, keo_png, column_width=column_width)
            if ok_keo:
                print(f"    ✓ keogram:   {keo_png.name}")
            else:
                print(f"    ✗ keogram failed: {keo_png.name}", file=sys.stderr)

            # 3) startrails
            trails_png = win_dir / "startrails.png"
            ok_trails = build_startrails(window_frames, combined_mask, trails_png, gamma=trails_gamma)
            if ok_trails:
                print(f"    ✓ trails:    {trails_png.name}")
            else:
                print(f"    ✗ trails failed: {trails_png.name}", file=sys.stderr)

            # 4) annotated window clip (if annotated frames exist)
            if has_annotated_frames:
                annotated_paths = [annotated_dir / f.name for f in window_frames]
                annotated_paths = [p for p in annotated_paths if p.exists()]
                if annotated_paths:
                    out_ann = win_dir / "timelapse_annotated_window.mp4"
                    ok_ann = build_timelapse_from_list(annotated_paths, out_ann, fps=fps)
                    if ok_ann:
                        print(f"    ✓ annotated: {out_ann.name}")
                    else:
                        print(f"    ✗ annotated timelapse failed: {out_ann.name}", file=sys.stderr)

    # ------------------------------------------------------------
    # FULL-NIGHT TIMELAPSE (only when --timelapse enabled)
    # ------------------------------------------------------------
    if timelapse:
        print("\n" + "="*60)
        print("Generating full-night timelapse videos...")
        print("="*60)

        timelapse_cfg = config.get("timelapse", {})
        fps = int(timelapse_cfg.get("fps", 30))
        quality = int(timelapse_cfg.get("quality", 8))

        timelapse_dir = ensure_dir(night / "timelapse")
        print(f"  Timelapse output directory: {timelapse_dir}")

        # 1) Raw frames timelapse (from frames/)
        raw_mp4 = timelapse_dir / "timelapse_raw.mp4"
        if build_timelapse_video(frames_dir, raw_mp4, fps=fps, quality=quality):
            print(f"  ✓ Raw timelapse saved to: {raw_mp4}")
        else:
            print("  ✗ Failed to create raw timelapse", file=sys.stderr)

        # 2) Annotated timelapse (if annotated frames exist)
        has_annotated_frames = False
        if annotated_dir.exists():
            has_annotated_frames = any(chain(
                annotated_dir.glob("*.png"),
                annotated_dir.glob("*.jpg"),
                annotated_dir.glob("*.jpeg")
            ))

        if has_annotated_frames:
            annotated_mp4 = timelapse_dir / "timelapse_annotated.mp4"
            if build_timelapse_video(annotated_dir, annotated_mp4, fps=fps, quality=quality):
                print(f"  ✓ Annotated timelapse saved to: {annotated_mp4}")
            else:
                print("  ✗ Failed to create annotated timelapse", file=sys.stderr)
        else:
            print("  Note: No annotated frames found (run with --overlay to generate them).")
            print(f"        Command: python analyze.py {night_dir} --overlay --timelapse")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze astronomical time-lapse frames for transient events (satellites, meteors, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with interest-based windows (default)
  python analyze.py ./night_2025-12-24
  
  # Use ML-based windows
  python analyze.py ./night_2025-12-24 --windows-source ml
  
  # Use hybrid windows (ML + non-overlapping interest)
  python analyze.py ./night_2025-12-24 --windows-source hybrid
  
  # With custom config and all tools
  python analyze.py ./night_2025-12-24 --config custom_config.yaml --all-tools
  python analyze.py ./night_2025-12-24 --validate --visualize --overlay
  
Output files:
  - masks/: sky_mask.png, persistent_edges.png, combined_mask.png
  - data/: metrics.csv, events.json, activity_windows.json, ml_windows.json (if ML mode)
  - activity/window_XX_YYYY_ZZZZ/: Per-window artifacts (timelapse, keogram, startrails)
  - plots/ (with --visualize): Time-series plots of metrics
  - annotated/ (with --overlay): Frames with detected streaks drawn
  - timelapse/: timelapse_annotated.mp4
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
    parser.add_argument(
        '--windows-source',
        choices=['interest', 'ml', 'hybrid'],
        default='interest',
        help='Window source: interest (default), ml, or hybrid. '
             'ML mode requires ml_windows.json from ML classifier. '
             'Hybrid merges both sources with IoU-based deduplication.'
    )
    
    args = parser.parse_args()
    
    # --all-tools flag enables all optional tools
    if args.all_tools:
        args.validate = True
        args.visualize = True
        args.overlay = True
        args.timelapse = True
    
    main(args.night_dir, args.config, args.validate, args.visualize, args.overlay, args.timelapse, args.windows_source)