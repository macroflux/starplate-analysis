#!/usr/bin/env python3
"""
Download images from allsky.local for a given date.
Usage: python download_images.py 20251219
       python download_images.py 20251219 --retry
"""

import sys
import os
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json

def format_date(date_str):
    """Convert date from 20251219 to 2025-12-19 format."""
    if len(date_str) != 8:
        raise ValueError("Date must be in YYYYMMDD format (e.g., 20251219)")
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

def get_failed_log_path(night_dir):
    """Get the path to the failed downloads log file."""
    return night_dir / "failed_downloads.json"

def load_failed_downloads(log_path):
    """Load the list of failed downloads from the log file."""
    if log_path.exists():
        with open(log_path, 'r') as f:
            return json.load(f)
    return []

def save_failed_downloads(log_path, failed_list):
    """Save the list of failed downloads to the log file."""
    with open(log_path, 'w') as f:
        json.dump(failed_list, f, indent=2)
    if failed_list:
        print(f"\n⚠ {len(failed_list)} failed downloads logged to: {log_path}")
    elif log_path.exists():
        # Remove the log file if all downloads succeeded
        log_path.unlink()
        print(f"\n✓ All downloads successful! Removed failed downloads log.")

def download_single_image(img_url, img_path, img_name, index, total):
    """Download a single image and return success status."""
    try:
        print(f"[{index}/{total}] Downloading {img_name}...", end=' ')
        img_response = requests.get(img_url, timeout=30)
        img_response.raise_for_status()
        
        with open(img_path, 'wb') as f:
            f.write(img_response.content)
        print("✓")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def download_images(date_str, retry_mode=False):
    """Download all images for the given date."""
    # Format the date for directory name
    formatted_date = format_date(date_str)
    
    # Create directory structure
    script_dir = Path(__file__).parent
    night_dir = script_dir / f"night_{formatted_date}"
    frames_dir = night_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory: {frames_dir}")
    
    # Get log file path
    log_path = get_failed_log_path(night_dir)
    
    # Construct the URL
    base_url = f"http://allsky.local/images/{date_str}/"
    
    # Determine which images to download
    if retry_mode:
        # Retry mode: load failed downloads
        failed_downloads = load_failed_downloads(log_path)
        if not failed_downloads:
            print("No failed downloads to retry!")
            return
        
        print(f"Retrying {len(failed_downloads)} failed downloads...")
        images = [item['filename'] for item in failed_downloads]
    else:
        # Normal mode: fetch directory listing
        try:
            print(f"Fetching directory listing from {base_url}")
            response = requests.get(base_url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML to find image links
            soup = BeautifulSoup(response.text, 'html.parser')
            image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
            
            images = []
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and any(href.lower().endswith(ext.lower()) for ext in image_extensions):
                    images.append(href)
            
            if not images:
                print("No images found in the directory listing.")
                return
            
            print(f"Found {len(images)} images to download")
        except requests.exceptions.RequestException as e:
            print(f"Error accessing {base_url}: {e}")
            sys.exit(1)
    
    # Download images and track failures
    failed_downloads = []
    for i, img_name in enumerate(images, 1):
        img_url = urljoin(base_url, img_name)
        img_path = frames_dir / img_name
        
        success = download_single_image(img_url, img_path, img_name, i, len(images))
        
        if not success:
            failed_downloads.append({
                'filename': img_name,
                'url': img_url
            })
    
    # Save failed downloads log
    save_failed_downloads(log_path, failed_downloads)
    
    # Print summary
    successful = len(images) - len(failed_downloads)
    print(f"\nDownload complete!")
    print(f"✓ {successful}/{len(images)} images downloaded successfully")
    if failed_downloads:
        print(f"✗ {len(failed_downloads)} images failed")
        print(f"\nTo retry failed downloads, run:")
        print(f"  python download_images.py {date_str} --retry")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_images.py YYYYMMDD [--retry]")
        print("Example: python download_images.py 20251219")
        print("         python download_images.py 20251219 --retry")
        sys.exit(1)
    
    date_input = sys.argv[1]
    retry_mode = len(sys.argv) > 2 and sys.argv[2] == '--retry'
    
    download_images(date_input, retry_mode)
