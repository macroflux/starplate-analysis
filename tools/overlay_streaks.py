#!/usr/bin/env python3
"""
Overlay detected streaks on original images.

This script reads the events.json file and draws detected streak lines
on the corresponding frame images, helping verify detection accuracy.

Usage:
    python tools/overlay_streaks.py <night_dir> [--output <output_dir>]

Examples:
    python tools/overlay_streaks.py ./night_2025-12-24
    python tools/overlay_streaks.py ./night_2025-12-24 --output ./custom_annotated
"""

import argparse
import json
from pathlib import Path
import sys
import cv2


def load_events(events_path: Path) -> list:
    """
    Load events from JSON file.
    
    Args:
        events_path: Path to events.json file
        
    Returns:
        List of event dictionaries
    """
    if not events_path.exists():
        print(f"Error: Events file not found: {events_path}", file=sys.stderr)
        print("Run analyze1.py first to generate events.", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(events_path, 'r') as f:
            events = json.load(f)
    except Exception as e:
        print(f"Error reading events file: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not events:
        print("No events found in events.json. No frames have detected streaks.", file=sys.stderr)
        print("Try adjusting detection parameters in config.yaml for more sensitive detection.")
        sys.exit(0)
    
    return events


def overlay_streaks(frame_path: Path, streaks: list, output_path: Path, 
                   line_color=(0, 0, 255), line_thickness=2):
    """
    Draw streak lines on an image and save the result.
    
    Args:
        frame_path: Path to original frame image
        streaks: List of streak coordinates [[x1,y1,x2,y2], ...]
        output_path: Path to save annotated image
        line_color: BGR color tuple for lines (default: red)
        line_thickness: Thickness of drawn lines in pixels
    """
    # Read image
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"Warning: Could not read image {frame_path}", file=sys.stderr)
        return False
    
    # Draw each streak
    for streak in streaks:
        x1, y1, x2, y2 = streak
        cv2.line(img, (x1, y1), (x2, y2), line_color, line_thickness)
    
    # Add text annotation
    text = f"Streaks: {len(streaks)}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, line_color, 2, cv2.LINE_AA)
    
    # Save annotated image
    cv2.imwrite(str(output_path), img)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Draw detected streaks on frame images from starplate-analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/overlay_streaks.py ./night_2025-12-24
  python tools/overlay_streaks.py ./night_2025-12-24 --output ./my_annotated
  python tools/overlay_streaks.py ./night_2025-12-24 --color 255,0,0 --thickness 3
  
Output:
  Creates annotated versions of frames with detected events,
  showing red lines where streaks were detected.
  Default output directory: <night_dir>/annotated/
"""
    )
    
    parser.add_argument(
        'night_dir',
        help='Path to night directory containing events.json and frames/'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output directory for annotated images (default: <night_dir>/annotated/)',
        default=None
    )
    parser.add_argument(
        '-c', '--color',
        help='Line color as R,G,B (e.g., "255,0,0" for red)',
        default='0,0,255'  # BGR format: red
    )
    parser.add_argument(
        '-t', '--thickness',
        help='Line thickness in pixels (default: 2)',
        type=int,
        default=2
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    night_dir = Path(args.night_dir)
    if not night_dir.exists():
        print(f"Error: Directory '{args.night_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    frames_dir = night_dir / 'frames'
    if not frames_dir.exists():
        print(f"Error: Frames directory '{frames_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Parse color
    try:
        color_parts = [int(c.strip()) for c in args.color.split(',')]
        if len(color_parts) != 3:
            raise ValueError
        # Convert from user-provided RGB format to OpenCV's BGR format
        # User provides "255,0,0" for red, converted to (0,0,255) for OpenCV
        line_color = (color_parts[2], color_parts[1], color_parts[0])
    except:
        print(f"Error: Invalid color format '{args.color}'. Use R,G,B format (e.g., '255,0,0')", 
              file=sys.stderr)
        sys.exit(1)
    
    # Load events
    events_path = night_dir / 'events.json'
    print(f"Loading events from {events_path}...")
    events = load_events(events_path)
    print(f"Found {len(events)} events to annotate")
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = night_dir / 'annotated'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving annotated images to {output_dir}...")
    
    # Process each event
    success_count = 0
    for event in events:
        filename = event['file']
        streaks = event['streaks']
        
        frame_path = frames_dir / filename
        if not frame_path.exists():
            print(f"Warning: Frame not found: {frame_path}", file=sys.stderr)
            continue
        
        output_path = output_dir / f"annotated_{filename}"
        
        if overlay_streaks(frame_path, streaks, output_path, 
                          line_color, args.thickness):
            success_count += 1
            print(f"  Processed: {filename} ({len(streaks)} streaks)")
    
    print(f"\nAnnotation complete!")
    print(f"Successfully annotated {success_count}/{len(events)} event frames")
    print(f"Output directory: {output_dir}")
    
    if success_count < len(events):
        print(f"Warning: {len(events) - success_count} frames could not be processed", 
              file=sys.stderr)


if __name__ == '__main__':
    main()
