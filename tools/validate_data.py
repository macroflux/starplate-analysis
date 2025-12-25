#!/usr/bin/env python3
"""
Validate data structure before running analysis.

This script checks that a night directory has the correct structure
and contains valid image files, helping identify issues before analysis.

Usage:
    python tools/validate_data.py <night_dir>

Examples:
    python tools/validate_data.py ./night_2025-12-24
"""

import argparse
from pathlib import Path
import sys
import cv2


def validate_directory_structure(night_dir: Path) -> bool:
    """
    Check if directory structure is correct.
    
    Args:
        night_dir: Path to night directory
        
    Returns:
        True if structure is valid, False otherwise
    """
    print("\n=== Directory Structure Validation ===")
    
    is_valid = True
    
    # Check if night directory exists
    if not night_dir.exists():
        print(f"❌ FAIL: Directory does not exist: {night_dir}")
        return False
    else:
        print(f"✓ Night directory exists: {night_dir}")
    
    # Check if frames subdirectory exists
    frames_dir = night_dir / 'frames'
    if not frames_dir.exists():
        print(f"❌ FAIL: frames/ subdirectory not found")
        print(f"   Expected: {frames_dir}")
        print(f"   Action: Create the directory and place .jpg files in it")
        is_valid = False
    else:
        print(f"✓ frames/ subdirectory exists")
    
    # Check for existing outputs (informational)
    mask_path = night_dir / 'sky_mask.png'
    metrics_path = night_dir / 'metrics.csv'
    events_path = night_dir / 'events.json'
    
    if mask_path.exists():
        print(f"ℹ sky_mask.png already exists (will be reused)")
    else:
        print(f"ℹ sky_mask.png not found (will be generated)")
    
    if metrics_path.exists():
        print(f"ℹ metrics.csv already exists (will be overwritten)")
    
    if events_path.exists():
        print(f"ℹ events.json already exists (will be overwritten)")
    
    return is_valid


def validate_frames(night_dir: Path) -> bool:
    """
    Validate image files in frames directory.
    
    Args:
        night_dir: Path to night directory
        
    Returns:
        True if frames are valid, False otherwise
    """
    print("\n=== Frame Images Validation ===")
    
    frames_dir = night_dir / 'frames'
    if not frames_dir.exists():
        return False
    
    # Find all .jpg files
    jpg_files = sorted(frames_dir.glob('*.jpg'))
    
    if not jpg_files:
        print(f"❌ FAIL: No .jpg files found in {frames_dir}")
        print(f"   Action: Add .jpg image files to the frames/ directory")
        
        # Check for other image formats
        other_formats = []
        for ext in ['*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            other_formats.extend(frames_dir.glob(ext))
        
        if other_formats:
            print(f"⚠ Warning: Found {len(other_formats)} images with other extensions:")
            for f in other_formats[:5]:
                print(f"   - {f.name}")
            if len(other_formats) > 5:
                print(f"   ... and {len(other_formats) - 5} more")
            print(f"   Note: analyze1.py only processes .jpg files")
        
        return False
    
    print(f"✓ Found {len(jpg_files)} .jpg files")
    
    # Sample a few images to check validity (limit to 5 for efficiency)
    sample_count = min(5, len(jpg_files))
    print(f"\nSampling {sample_count} images for validation...")
    
    valid_count = 0
    dimensions = set()
    
    for i, frame_path in enumerate(jpg_files[:sample_count]):
        # Load image with IMREAD_UNCHANGED to preserve original bit depth and channels
        # This ensures we can detect any corruption without modifying the image data
        img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"❌ FAIL: Cannot read {frame_path.name}")
        else:
            h, w = img.shape[:2]
            dimensions.add((w, h))
            valid_count += 1
            if i == 0:
                print(f"✓ First image: {frame_path.name}")
                print(f"  Dimensions: {w}x{h} pixels")
            # Free memory immediately
            del img
    
    if valid_count < sample_count:
        print(f"⚠ Warning: {sample_count - valid_count}/{sample_count} sample images could not be read")
        return False
    else:
        print(f"✓ All sampled images are readable")
    
    # Check dimension consistency
    if len(dimensions) > 1:
        print(f"⚠ Warning: Images have inconsistent dimensions:")
        for dim in dimensions:
            print(f"   - {dim[0]}x{dim[1]} pixels")
        print(f"   This may cause issues during analysis")
    else:
        w, h = list(dimensions)[0]
        print(f"✓ Image dimensions are consistent: {w}x{h} pixels")
    
    return True


def validate_config(night_dir: Path) -> bool:
    """
    Check for configuration file.
    
    Args:
        night_dir: Path to night directory
        
    Returns:
        True (informational only)
    """
    print("\n=== Configuration Validation ===")
    
    # Check for config.yaml in current directory
    config_paths = [
        Path('config.yaml'),
        night_dir / 'config.yaml',
        night_dir.parent / 'config.yaml'
    ]
    
    found_config = None
    for config_path in config_paths:
        if config_path.exists():
            found_config = config_path
            break
    
    if found_config:
        print(f"✓ Configuration file found: {found_config}")
        print(f"  Use --config flag to specify a different configuration")
    else:
        print(f"ℹ No config.yaml found (will use default parameters)")
        print(f"  Create config.yaml to customize detection parameters")
    
    return True


def print_summary(night_dir: Path, all_valid: bool):
    """
    Print validation summary and next steps.
    """
    print("\n" + "=" * 60)
    
    if all_valid:
        print("✓ VALIDATION PASSED")
        print("\nYour data structure is ready for analysis!")
        print("\nNext steps:")
        print(f"  1. Run analysis:")
        print(f"     python prep1/analyze1.py {night_dir}")
        print(f"\n  2. Visualize results:")
        print(f"     python tools/visualize.py {night_dir}")
        print(f"\n  3. Annotate streaks:")
        print(f"     python tools/overlay_streaks.py {night_dir}")
    else:
        print("❌ VALIDATION FAILED")
        print("\nPlease fix the issues above before running analysis.")
        print("\nExpected structure:")
        print(f"  {night_dir}/")
        print(f"  ├── frames/")
        print(f"  │   ├── image001.jpg")
        print(f"  │   ├── image002.jpg")
        print(f"  │   └── ...")
        print(f"  └── (outputs will be generated here)")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate data structure for starplate-analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/validate_data.py ./night_2025-12-24
  
This tool checks:
  - Directory structure (night_dir/frames/ exists)
  - Image files exist and are readable
  - Image format and dimensions
  - Configuration file presence
  
Run this before analyze1.py to catch issues early.
"""
    )
    
    parser.add_argument(
        'night_dir',
        help='Path to night directory to validate'
    )
    
    args = parser.parse_args()
    
    night_dir = Path(args.night_dir)
    
    print("=" * 60)
    print("starplate-analysis Data Validation")
    print("=" * 60)
    print(f"\nValidating: {night_dir.absolute()}")
    
    # Run all validations
    structure_ok = validate_directory_structure(night_dir)
    frames_ok = validate_frames(night_dir) if structure_ok else False
    config_ok = validate_config(night_dir)
    
    all_valid = structure_ok and frames_ok
    
    # Print summary
    print_summary(night_dir, all_valid)
    
    # Exit with appropriate code
    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
