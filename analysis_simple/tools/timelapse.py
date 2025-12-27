#!/usr/bin/env python3
"""
Timelapse Video Generator

Generates MP4 timelapse videos from image sequences using OpenCV.
Can be used standalone or integrated into analysis pipeline.

Usage:
    python timelapse.py <frames_dir> [options]
    python timelapse.py <frames_dir> --output video.mp4 --fps 30 --quality 8
"""

import argparse
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Error: opencv-python not installed", file=sys.stderr)
    print("Install with: pip install opencv-python", file=sys.stderr)
    sys.exit(1)


def build_timelapse(
    frames_dir: Path,
    output_path: Path,
    fps: int = 30,
    pattern: str = "*.jpg",
    quality: int = 8,
    verbose: bool = True
) -> bool:
    """
    Build an MP4 timelapse from image frames using OpenCV.
    
    Args:
        frames_dir: Directory containing image frames
        output_path: Output MP4 path
        fps: Frames per second (default 30)
        pattern: Glob pattern for frames (default *.jpg)
        quality: Video quality parameter (currently not applied to encoding; reserved for future CRF mapping)
        verbose: Print progress messages (default True)
        
    Returns:
        True if successful, False otherwise
    """
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}", file=sys.stderr)
        return False
    
    frames = sorted(frames_dir.glob(pattern))
    if not frames:
        print(f"Error: No frames found in {frames_dir} matching {pattern}", file=sys.stderr)
        return False
    
    if verbose:
        print(f"Building MP4 timelapse")
        print(f"  Input: {frames_dir}")
        print(f"  Pattern: {pattern}")
        print(f"  Frames: {len(frames)}")
        print(f"  FPS: {fps}")
        print(f"  Quality: {quality}")
        print(f"  Output: {output_path}")
        print()
    
    try:
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frames[0]))
        if first_frame is None:
            print(f"Error: Could not read first frame: {frames[0]}", file=sys.stderr)
            return False
        
        height, width = first_frame.shape[:2]
        
        # Note: quality parameter is reserved for future CRF mapping
        # Currently using codec defaults with VideoWriter
        
        # Create VideoWriter with H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print(f"Error: Could not create video writer", file=sys.stderr)
            return False
        
        # Write frames
        for i, frame_path in enumerate(frames, 1):
            if verbose and (i % 50 == 0 or i == len(frames)):
                progress = 100 * i // len(frames)
                print(f"  Encoding: {i}/{len(frames)} ({progress}%)")
            
            img = cv2.imread(str(frame_path))
            if img is not None:
                writer.write(img)
            else:
                print(f"Warning: Skipping unreadable frame: {frame_path}", file=sys.stderr)
        
        writer.release()
        
        if verbose:
            print()
            print(f"✓ Timelapse saved: {output_path}")
            
        return True
        
    except Exception as e:
        print(f"Error creating timelapse: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate MP4 timelapse videos from image sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create timelapse from frames directory (default: timelapse.mp4)
  python timelapse.py path/to/frames

  # Specify output file and settings
  python timelapse.py path/to/frames --output night.mp4 --fps 25

  # High quality timelapse with custom pattern
  python timelapse.py path/to/frames --pattern "*.png" --quality 5 --fps 60

  # Create from annotated images
  python timelapse.py data/night_2025-12-24/annotated --output annotated.mp4
        """
    )
    
    parser.add_argument(
        'frames_dir',
        type=Path,
        help='Directory containing image frames'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output MP4 path (default: frames_dir/timelapse.mp4)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second (default: 30)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.jpg',
        help='Glob pattern for frame files (default: *.jpg)'
    )
    
    parser.add_argument(
        '--quality',
        type=int,
        default=8,
        choices=range(0, 11),
        help='Video quality 0-10, lower=higher (default: 8)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Set default output path if not specified
    if args.output is None:
        args.output = args.frames_dir / "timelapse.mp4"
    
    # Build the timelapse
    success = build_timelapse(
        frames_dir=args.frames_dir,
        output_path=args.output,
        fps=args.fps,
        pattern=args.pattern,
        quality=args.quality,
        verbose=not args.quiet
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
