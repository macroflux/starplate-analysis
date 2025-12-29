#!/usr/bin/env python3
"""
ML Window Inference

Turns per-frame ML activity probabilities into contiguous activity windows.

Inputs (expected):
  <night>/ml/predictions.csv
  <night>/frames/*.jpg
  <night>/masks/combined_mask.png  (optional for artifacts)

Outputs:
  <night>/data/ml_windows.json
  <night>/ml/predictions_smoothed.csv  (optional convenience output)

Usage:
  python analysis_ml_windows/infer_windows.py data/night_2025-12-27
  python analysis_ml_windows/infer_windows.py data/night_2025-12-27 --artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import cv2
except ImportError:
    print("Error: opencv-python not installed. pip install opencv-python", file=sys.stderr)
    sys.exit(1)


# -------------------------
# Config / IO helpers
# -------------------------

def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        r = csv.DictReader(fp)
        return list(r)


def write_csv_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def find_probability_column(headers: List[str]) -> Optional[str]:
    """
    Try to locate an activity probability column in predictions.csv robustly.

    Common names to match:
      prob_activity, p_activity, activity_prob, proba, probability, y_prob, pred_prob
    """
    h = [x.strip() for x in headers]

    # exact-ish preferred matches
    preferred = [
        "prob_activity",
        "p_activity",
        "activity_prob",
        "activity_probability",
        "probability",
        "proba",
        "y_prob",
        "pred_prob",
        "p",
    ]
    lower_map = {name.lower(): name for name in h}

    for key in preferred:
        if key in lower_map:
            return lower_map[key]

    # fallback: any column containing these tokens
    tokens = ["prob", "proba", "probability"]
    for name in h:
        ln = name.lower()
        if any(t in ln for t in tokens):
            return name

    return None


# -------------------------
# Smoothing + windowing
# -------------------------

def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    if x.size == 0:
        return x
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    w = int(window)
    if w <= 1 or x.size == 0:
        return x.astype(np.float64)
    w = max(1, w)
    # pad edges
    pad = w // 2
    xp = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / float(w)
    y = np.convolve(xp, kernel, mode="valid")
    return y


def detect_windows_from_signal(
    signal: np.ndarray,
    threshold: float,
    min_len: int,
    pad: int,
    merge_gap: int,
    max_windows: int,
) -> List[dict]:
    """
    Convert a 1D signal (probabilities) into merged/padded windows where signal >= threshold.
    """
    n = int(signal.size)
    if n == 0:
        return []

    active = signal >= float(threshold)
    idx = np.where(active)[0]
    if idx.size == 0:
        return []

    # contiguous runs
    runs: List[Tuple[int, int]] = []
    s = int(idx[0])
    prev = int(idx[0])
    for k in idx[1:]:
        k = int(k)
        if k == prev + 1:
            prev = k
        else:
            runs.append((s, prev))
            s = k
            prev = k
    runs.append((s, prev))

    # min_len + padding
    padded: List[Tuple[int, int]] = []
    for s, e in runs:
        if (e - s + 1) < int(min_len):
            continue
        ps = max(0, s - int(pad))
        pe = min(n - 1, e + int(pad))
        padded.append((ps, pe))

    if not padded:
        return []

    # merge close
    merged = [padded[0]]
    for s, e in padded[1:]:
        ms, me = merged[-1]
        if s <= me + int(merge_gap):
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))

    # build window dicts
    windows: List[dict] = []
    for s, e in merged:
        seg = signal[s : e + 1]
        peak_rel = int(np.argmax(seg))
        peak_idx = s + peak_rel
        windows.append(
            {
                "start": int(s),
                "end": int(e),
                "peak_index": int(peak_idx),
                "peak_value": float(signal[peak_idx]),
                "length": int(e - s + 1),
                "score_sum": float(np.sum(seg)),
                "threshold": float(threshold),
                "source": "ml",
            }
        )

    # rank by score_sum then peak (more stable than peak only)
    windows.sort(key=lambda w: (w["score_sum"], w["peak_value"]), reverse=True)
    windows = windows[: int(max_windows)]
    windows.sort(key=lambda w: w["start"])
    return windows


# -------------------------
# Optional artifact generation
# -------------------------

def load_gray(path: Path, target_shape=None) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if target_shape is not None and g.shape != target_shape:
        return None
    return g


def build_keogram(frames: List[Path], mask: np.ndarray, out_png: Path, column_width: int = 3) -> bool:
    h, w = mask.shape
    cx = w // 2
    half = max(1, int(column_width) // 2)

    strips = []
    for p in frames:
        g = load_gray(p, target_shape=(h, w))
        if g is None:
            continue
        gm = cv2.bitwise_and(g, g, mask=mask)
        x0 = max(0, cx - half)
        x1 = min(w, cx + half + 1)
        strip = gm[:, x0:x1]
        col = np.mean(strip.astype(np.float32), axis=1).astype(np.uint8)
        strips.append(col)

    if len(strips) < 2:
        return False

    keo = np.stack(strips, axis=1)
    ensure_dir(out_png.parent)
    cv2.imwrite(str(out_png), keo)
    return True


def build_startrails(frames: List[Path], mask: np.ndarray, out_png: Path, gamma: float = 1.0) -> bool:
    h, w = mask.shape
    acc = np.zeros((h, w), dtype=np.uint8)
    used = 0

    for p in frames:
        g = load_gray(p, target_shape=(h, w))
        if g is None:
            continue
        gm = cv2.bitwise_and(g, g, mask=mask)
        acc = np.maximum(acc, gm)
        used += 1

    if used < 2:
        return False

    if gamma and abs(float(gamma) - 1.0) > 1e-3:
        f = acc.astype(np.float32) / 255.0
        f = np.power(f, 1.0 / float(gamma))
        acc = np.clip(f * 255.0, 0, 255).astype(np.uint8)

    ensure_dir(out_png.parent)
    cv2.imwrite(str(out_png), acc)
    return True


def build_timelapse_from_list(frame_paths: List[Path], out_mp4: Path, fps: int = 30) -> bool:
    if not frame_paths:
        return False
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return False
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, int(fps), (w, h))
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


# -------------------------
# Main
# -------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Infer activity windows from ML per-frame probabilities")
    parser.add_argument("night_dir", type=Path, help="Night directory, e.g. data/night_2025-12-27")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML path (default: script_dir/config.yaml)")
    parser.add_argument("--artifacts", action="store_true", help="Generate per-window artifacts under night/activity_ml/")
    parser.add_argument("--write-smoothed", action="store_true", help="Write ml/predictions_smoothed.csv")
    args = parser.parse_args()

    night = args.night_dir
    if not night.exists():
        print(f"Error: night_dir not found: {night}", file=sys.stderr)
        return 2

    # Default to config.yaml in the same directory as this script
    if args.config is None:
        script_dir = Path(__file__).parent
        args.config = script_dir / "config.yaml"
    
    cfg = {}
    if args.config.exists():
        cfg = load_yaml(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        print(f"Warning: Config not found at {args.config}, using defaults", file=sys.stderr)
    mw = cfg.get("ml_windows", {})

    prob_threshold = float(mw.get("prob_threshold", 0.65))
    min_len = int(mw.get("min_len", 6))
    pad = int(mw.get("pad", 10))
    merge_gap = int(mw.get("merge_gap", 8))
    max_windows = int(mw.get("max_windows", 10))

    smoothing = str(mw.get("smoothing", "ema")).lower()
    ema_alpha = float(mw.get("ema_alpha", 0.20))
    ma_window = int(mw.get("ma_window", 9))

    fps = int(mw.get("fps", 30))
    keo_w = int(mw.get("keogram_column_width", 3))
    trails_gamma = float(mw.get("startrails_gamma", 1.2))

    pred_csv = night / "ml" / "predictions.csv"
    if not pred_csv.exists():
        print(f"Error: predictions.csv not found at {pred_csv}", file=sys.stderr)
        print("Run classifier first: python analysis_ml_activity_classifier/train.py <night_dir>", file=sys.stderr)
        return 2

    rows = read_csv_rows(pred_csv)
    if not rows:
        print(f"Error: predictions.csv is empty: {pred_csv}", file=sys.stderr)
        return 2

    headers = list(rows[0].keys())
    prob_col = find_probability_column(headers)
    file_col = "file" if "file" in rows[0] else ("filename" if "filename" in rows[0] else None)
    if file_col is None:
        print(f"Error: predictions.csv must contain a 'file' (or 'filename') column.", file=sys.stderr)
        return 2
    if prob_col is None:
        print(f"Error: couldn't find probability column in predictions.csv headers: {headers}", file=sys.stderr)
        return 2

    files = [r[file_col] for r in rows]
    probs = np.array([float(r[prob_col]) for r in rows], dtype=np.float64)

    if smoothing == "ema":
        smooth = ema(probs, alpha=ema_alpha)
    elif smoothing in ("ma", "moving_average", "moving-average"):
        smooth = moving_average(probs, window=ma_window)
    else:
        print(f"Warning: unknown smoothing '{smoothing}', using raw probs")
        smooth = probs.astype(np.float64)

    windows = detect_windows_from_signal(
        smooth,
        threshold=prob_threshold,
        min_len=min_len,
        pad=pad,
        merge_gap=merge_gap,
        max_windows=max_windows,
    )

    out_data = ensure_dir(night / "data")
    out_windows = out_data / "ml_windows.json"
    out_windows.write_text(json.dumps(windows, indent=2), encoding="utf-8")

    print("\n============================================================")
    print("ML Window Inference")
    print("============================================================")
    print(f"Input:  {pred_csv}")
    print(f"Prob:   column='{prob_col}' smoothing='{smoothing}'")
    print(f"Output: {out_windows}")
    print(f"Found {len(windows)} windows (threshold={prob_threshold})")

    if windows:
        for i, w in enumerate(windows):
            print(
                f"  {i:02d}: frames {w['start']}–{w['end']} "
                f"(len={w['length']}) peak@{w['peak_index']}={w['peak_value']:.3f} sum={w['score_sum']:.2f}"
            )

    if args.write_smoothed:
        sm_rows: List[Dict[str, object]] = []
        for i, r in enumerate(rows):
            rr: Dict[str, object] = dict(r)
            rr["prob_smoothed"] = str(float(smooth[i]))
            sm_rows.append(rr)
        out_sm = night / "ml" / "predictions_smoothed.csv"
        write_csv_rows(out_sm, sm_rows)
        print(f"\nWrote: {out_sm}")

    # Optional artifacts
    if args.artifacts and windows:
        frames_dir = night / "frames"
        frames = sorted(frames_dir.glob("*.jpg"))
        if not frames:
            print(f"\nNote: No frames found at {frames_dir}; skipping artifacts.")
            return 0

        mask_path = night / "masks" / "combined_mask.png"
        mask = None
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"\nNote: combined mask not found/readable at {mask_path}; artifacts will be limited.")
            # For keogram/startrails, we really want a mask. We'll skip those if missing.

        activity_root = ensure_dir(night / "activity_ml")

        for wi, w in enumerate(windows):
            s, e = int(w["start"]), int(w["end"])
            window_frames = frames[s : e + 1]
            win_dir = ensure_dir(activity_root / f"window_{wi:02d}_{s:04d}_{e:04d}")

            # timelapse (raw)
            out_mp4 = win_dir / "timelapse_window.mp4"
            ok_vid = build_timelapse_from_list(window_frames, out_mp4, fps=fps)

            # keogram/startrails only if mask exists
            ok_keo = False
            ok_trails = False
            if mask is not None:
                ok_keo = build_keogram(window_frames, mask, win_dir / "keogram.png", column_width=keo_w)
                ok_trails = build_startrails(window_frames, mask, win_dir / "startrails.png", gamma=trails_gamma)

            # annotated window timelapse if annotated frames exist
            ann_dir = night / "annotated"
            ok_ann = False
            if ann_dir.exists():
                ann_paths = [ann_dir / p.name for p in window_frames]
                ann_paths = [p for p in ann_paths if p.exists()]
                if ann_paths:
                    ok_ann = build_timelapse_from_list(ann_paths, win_dir / "timelapse_annotated_window.mp4", fps=fps)

            print(f"\nArtifacts window {wi:02d} frames {s}-{e}:")
            print(f"  timelapse:  {'✓' if ok_vid else '✗'}")
            if mask is not None:
                print(f"  keogram:    {'✓' if ok_keo else '✗'}")
                print(f"  startrails: {'✓' if ok_trails else '✗'}")
            print(f"  annotated:  {'✓' if ok_ann else '-'}")

        print(f"\nArtifacts saved under: {activity_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
