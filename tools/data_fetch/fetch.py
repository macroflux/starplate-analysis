#!/usr/bin/env python3
"""
Image fetching tool for downloading astroplate images from allsky.local.

This module provides a robust, configurable way to download astronomical
time-lapse images with retry logic, progress tracking, and error handling.

Usage:
    As a script:
        python fetch.py 20251224
        python fetch.py 20251224 --verbose --retry
        python fetch.py 20251224 --end-date 20251226
    
    As a module:
        from fetch import ImageFetcher
        fetcher = ImageFetcher()
        fetcher.download_night("20251224")
"""

import sys
import os
import argparse
import logging
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urljoin

import requests
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm


class ImageFetcher:
    """
    Fetch astronomical images from allsky.local server.
    
    This class handles downloading images with retry logic, progress tracking,
    and comprehensive error handling.
    """
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        verbose: bool = False,
        quiet: bool = False
    ):
        """
        Initialize the ImageFetcher.
        
        Args:
            config_path: Path to YAML configuration file
            verbose: Enable verbose logging
            quiet: Enable quiet mode (errors only)
        """
        self.config = self._load_config(config_path)
        self._setup_logging(verbose, quiet)
        self.stats = {
            'total_size': 0,
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file with environment variable overrides.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {config_path} not found, using defaults")
            config = self._get_default_config()
        
        # Environment variable overrides
        if 'BASE_URL' in os.environ:
            config['base_url'] = os.environ['BASE_URL']
        if 'NETWORK_TIMEOUT' in os.environ:
            config['network']['timeout'] = int(os.environ['NETWORK_TIMEOUT'])
        if 'DATA_DIR' in os.environ:
            config['output']['data_dir'] = os.environ['DATA_DIR']
        
        return config
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'base_url': 'http://allsky.local/images/',
            'network': {
                'timeout': 30,
                'listing_timeout': 10,
                'max_retries': 3,
                'backoff_multiplier': 2,
                'initial_backoff': 1
            },
            'image_extensions': ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'],
            'download': {
                'verify_checksum': False,
                'chunk_size': 8192
            },
            'output': {
                'data_dir': '../../data',
                'night_dir_format': 'night_{date}',
                'frames_subdir': 'frames',
                'failed_log_filename': 'failed_downloads.json'
            }
        }
    
    def _setup_logging(self, verbose: bool, quiet: bool) -> None:
        """
        Configure logging based on verbosity level.
        
        Args:
            verbose: Enable verbose logging
            quiet: Enable quiet mode
        """
        if quiet:
            level = logging.ERROR
        elif verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_date_format(self, date_str: str) -> bool:
        """
        Validate date string format.
        
        Args:
            date_str: Date in YYYYMMDD format
            
        Returns:
            True if valid, False otherwise
        """
        if len(date_str) != 8:
            return False
        try:
            datetime.strptime(date_str, '%Y%m%d')
            return True
        except ValueError:
            return False
    
    def format_date(self, date_str: str) -> str:
        """
        Convert date from YYYYMMDD to YYYY-MM-DD format.
        
        Args:
            date_str: Date in YYYYMMDD format
            
        Returns:
            Date in YYYY-MM-DD format
            
        Raises:
            ValueError: If date format is invalid
        """
        if not self.validate_date_format(date_str):
            raise ValueError(f"Invalid date format: {date_str}. Expected YYYYMMDD")
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    
    def get_night_directory(self, date_str: str) -> Path:
        """
        Get the night directory path for a given date.
        
        Args:
            date_str: Date in YYYYMMDD format
            
        Returns:
            Path to night directory
        """
        formatted_date = self.format_date(date_str)
        script_dir = Path(__file__).parent
        data_dir = script_dir / self.config['output']['data_dir']
        night_dir_name = self.config['output']['night_dir_format'].format(date=formatted_date)
        return data_dir / night_dir_name
    
    def get_frames_directory(self, date_str: str) -> Path:
        """
        Get the frames directory path for a given date.
        
        Args:
            date_str: Date in YYYYMMDD format
            
        Returns:
            Path to frames directory
        """
        night_dir = self.get_night_directory(date_str)
        return night_dir / self.config['output']['frames_subdir']
    
    def get_failed_log_path(self, date_str: str) -> Path:
        """
        Get the path to the failed downloads log file.
        
        Args:
            date_str: Date in YYYYMMDD format
            
        Returns:
            Path to failed downloads log
        """
        night_dir = self.get_night_directory(date_str)
        return night_dir / self.config['output']['failed_log_filename']
    
    def load_failed_downloads(self, log_path: Path) -> List[Dict[str, str]]:
        """
        Load the list of failed downloads from log file.
        
        Args:
            log_path: Path to log file
            
        Returns:
            List of failed download records
        """
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse {log_path}")
                return []
        return []
    
    def save_failed_downloads(
        self,
        log_path: Path,
        failed_list: List[Dict[str, str]]
    ) -> None:
        """
        Save the list of failed downloads to log file.
        
        Args:
            log_path: Path to log file
            failed_list: List of failed download records
        """
        if failed_list:
            with open(log_path, 'w') as f:
                json.dump(failed_list, f, indent=2)
            self.logger.warning(f"{len(failed_list)} failed downloads logged to: {log_path}")
        elif log_path.exists():
            # Remove log file if all downloads succeeded
            log_path.unlink()
            self.logger.info("All downloads successful! Removed failed downloads log.")
    
    def calculate_checksum(self, file_path: Path) -> str:
        """
        Calculate MD5 checksum of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 checksum as hex string
        """
        md5 = hashlib.md5()
        chunk_size = self.config['download']['chunk_size']
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                md5.update(chunk)
        return md5.hexdigest()
    
    def fetch_directory_listing(
        self,
        date_str: str
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        Fetch directory listing from server with retry logic.
        
        Args:
            date_str: Date in YYYYMMDD format
            
        Returns:
            Tuple of (list of image filenames, error message)
        """
        base_url = self.config['base_url']
        url = f"{base_url}{date_str}/"
        
        timeout = self.config['network']['listing_timeout']
        max_retries = self.config['network']['max_retries']
        backoff = self.config['network']['initial_backoff']
        multiplier = self.config['network']['backoff_multiplier']
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Fetching directory listing from {url} (attempt {attempt + 1}/{max_retries})")
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                
                # Parse HTML to find image links
                soup = BeautifulSoup(response.text, 'html.parser')
                image_extensions = tuple(self.config['image_extensions'])
                
                images = []
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href and any(href.lower().endswith(ext.lower()) for ext in image_extensions):
                        images.append(href)
                
                self.logger.info(f"Found {len(images)} images for {date_str}")
                return images, None
                
            except requests.exceptions.Timeout:
                error = f"Timeout connecting to {url}"
                self.logger.warning(f"{error} (attempt {attempt + 1}/{max_retries})")
            except requests.exceptions.ConnectionError:
                error = f"Connection error to {url}"
                self.logger.warning(f"{error} (attempt {attempt + 1}/{max_retries})")
            except requests.exceptions.RequestException as e:
                error = f"Request failed: {e}"
                self.logger.warning(f"{error} (attempt {attempt + 1}/{max_retries})")
            
            # Exponential backoff
            if attempt < max_retries - 1:
                sleep_time = backoff * (multiplier ** attempt)
                self.logger.debug(f"Backing off for {sleep_time}s before retry")
                time.sleep(sleep_time)
        
        return None, error
    
    def download_single_image(
        self,
        img_url: str,
        img_path: Path,
        force: bool = False
    ) -> Tuple[bool, Optional[str], int, bool]:
        """
        Download a single image with retry logic.
        
        Args:
            img_url: URL of image to download
            img_path: Local path to save image
            force: Force re-download if file exists
            
        Returns:
            Tuple of (success, error_message, file_size, was_skipped)
        """
        # Skip if file exists and not forcing
        if img_path.exists() and not force:
            file_size = img_path.stat().st_size
            self.logger.debug(f"Skipping existing file: {img_path.name}")
            return True, None, file_size, True
        
        timeout = self.config['network']['timeout']
        max_retries = self.config['network']['max_retries']
        backoff = self.config['network']['initial_backoff']
        multiplier = self.config['network']['backoff_multiplier']
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Downloading {img_url} (attempt {attempt + 1}/{max_retries})")
                response = requests.get(img_url, timeout=timeout, stream=True)
                response.raise_for_status()
                
                # Get content length for progress
                total_size = int(response.headers.get('content-length', 0))
                
                # Download with progress
                with open(img_path, 'wb') as f:
                    if total_size == 0:
                        f.write(response.content)
                    else:
                        for chunk in response.iter_content(chunk_size=self.config['download']['chunk_size']):
                            if chunk:
                                f.write(chunk)
                
                file_size = img_path.stat().st_size
                self.logger.debug(f"Successfully downloaded {img_path.name} ({file_size} bytes)")
                return True, None, file_size, False
                
            except requests.exceptions.Timeout:
                error = "Timeout"
                self.logger.debug(f"{error} downloading {img_url} (attempt {attempt + 1}/{max_retries})")
            except requests.exceptions.ConnectionError:
                error = "Connection error"
                self.logger.debug(f"{error} downloading {img_url} (attempt {attempt + 1}/{max_retries})")
            except requests.exceptions.RequestException as e:
                error = str(e)
                self.logger.debug(f"Error downloading {img_url}: {error} (attempt {attempt + 1}/{max_retries})")
            except IOError as e:
                error = f"File write error: {e}"
                self.logger.error(error)
                return False, error, 0, False
            
            # Exponential backoff
            if attempt < max_retries - 1:
                sleep_time = backoff * (multiplier ** attempt)
                self.logger.debug(f"Backing off for {sleep_time}s before retry")
                time.sleep(sleep_time)
        
        return False, error, 0, False
    
    def download_night(
        self,
        date_str: str,
        retry_mode: bool = False,
        dry_run: bool = False,
        force: bool = False
    ) -> bool:
        """
        Download all images for a given night.
        
        Args:
            date_str: Date in YYYYMMDD format
            retry_mode: Retry only failed downloads
            dry_run: Preview without downloading
            force: Force re-download existing files
            
        Returns:
            True if all downloads successful, False otherwise
        """
        # Validate date
        try:
            formatted_date = self.format_date(date_str)
        except ValueError as e:
            self.logger.error(str(e))
            return False
        
        # Create directory structure
        frames_dir = self.get_frames_directory(date_str)
        frames_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {frames_dir}")
        
        # Get log file path
        log_path = self.get_failed_log_path(date_str)
        
        # Determine which images to download
        if retry_mode:
            # Retry mode: load failed downloads
            failed_downloads = self.load_failed_downloads(log_path)
            if not failed_downloads:
                self.logger.info("No failed downloads to retry!")
                return True
            
            self.logger.info(f"Retrying {len(failed_downloads)} failed downloads...")
            images = [item['filename'] for item in failed_downloads]
        else:
            # Normal mode: fetch directory listing
            images, error = self.fetch_directory_listing(date_str)
            if images is None:
                self.logger.error(f"Failed to fetch directory listing: {error}")
                return False
            
            if not images:
                self.logger.warning("No images found in directory listing")
                return True
        
        if dry_run:
            self.logger.info(f"DRY RUN: Would download {len(images)} images")
            for img in images[:10]:  # Show first 10
                self.logger.info(f"  - {img}")
            if len(images) > 10:
                self.logger.info(f"  ... and {len(images) - 10} more")
            return True
        
        # Download images with progress bar
        base_url = self.config['base_url']
        url_base = f"{base_url}{date_str}/"
        failed_downloads = []
        
        self.stats['start_time'] = time.time()
        self.stats['total_files'] = len(images)
        
        with tqdm(total=len(images), desc=f"Downloading {date_str}", unit="file") as pbar:
            for img_name in images:
                img_url = urljoin(url_base, img_name)
                img_path = frames_dir / img_name
                
                success, error, file_size, was_skipped = self.download_single_image(img_url, img_path, force)
                
                if success:
                    self.stats['successful'] += 1
                    self.stats['total_size'] += file_size
                    if was_skipped:
                        self.stats['skipped'] += 1
                else:
                    self.stats['failed'] += 1
                    failed_downloads.append({
                        'filename': img_name,
                        'url': img_url,
                        'error': error
                    })
                
                pbar.update(1)
        
        self.stats['end_time'] = time.time()
        
        # Save failed downloads log
        self.save_failed_downloads(log_path, failed_downloads)
        
        # Print summary
        self._print_summary(date_str, failed_downloads)
        
        return len(failed_downloads) == 0
    
    def _print_summary(self, date_str: str, failed_downloads: List[Dict]) -> None:
        """
        Print download summary statistics.
        
        Args:
            date_str: Date string
            failed_downloads: List of failed downloads
        """
        duration = self.stats['end_time'] - self.stats['start_time']
        total_mb = self.stats['total_size'] / (1024 * 1024)
        speed_mbps = (total_mb / duration) if duration > 0 else 0
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"Download Summary for {date_str}")
        self.logger.info("=" * 60)
        self.logger.info(f"Total files:      {self.stats['total_files']}")
        self.logger.info(f"Successful:       {self.stats['successful']}")
        self.logger.info(f"Failed:           {self.stats['failed']}")
        self.logger.info(f"Skipped:          {self.stats['skipped']} (already existed)")
        self.logger.info(f"Total size:       {total_mb:.2f} MB")
        self.logger.info(f"Duration:         {duration:.2f} seconds")
        self.logger.info(f"Average speed:    {speed_mbps:.2f} MB/s")
        self.logger.info("=" * 60)
        
        if failed_downloads:
            self.logger.warning(f"\nTo retry failed downloads, run:")
            self.logger.warning(f"  python fetch.py {date_str} --retry")
    
    def download_date_range(
        self,
        start_date: str,
        end_date: str,
        dry_run: bool = False,
        force: bool = False
    ) -> bool:
        """
        Download images for a date range.
        
        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            dry_run: Preview without downloading
            force: Force re-download existing files
            
        Returns:
            True if all downloads successful, False otherwise
        """
        try:
            start = datetime.strptime(start_date, '%Y%m%d')
            end = datetime.strptime(end_date, '%Y%m%d')
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            return False
        
        if start > end:
            self.logger.error("Start date must be before or equal to end date")
            return False
        
        # Generate date range
        current = start
        dates = []
        while current <= end:
            dates.append(current.strftime('%Y%m%d'))
            current += timedelta(days=1)
        
        self.logger.info(f"Downloading {len(dates)} nights: {dates[0]} to {dates[-1]}")
        
        all_success = True
        for date in dates:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Processing date: {date}")
            self.logger.info(f"{'=' * 60}")
            
            success = self.download_night(date, dry_run=dry_run, force=force)
            if not success:
                all_success = False
        
        return all_success
    
    def retry_failed(self, date_str: str) -> bool:
        """
        Retry failed downloads for a given date.
        
        Args:
            date_str: Date in YYYYMMDD format
            
        Returns:
            True if all retries successful, False otherwise
        """
        return self.download_night(date_str, retry_mode=True)


def main():
    """Command-line interface for the image fetcher."""
    parser = argparse.ArgumentParser(
        description="Download astroplate images from allsky.local",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 20251224
  %(prog)s 20251224 --verbose
  %(prog)s 20251224 --retry
  %(prog)s 20251224 --dry-run
  %(prog)s 20251224 --end-date 20251226
        """
    )
    
    parser.add_argument(
        'date',
        help='Date in YYYYMMDD format (e.g., 20251224)'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    
    # Verbosity options
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    verbosity.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet output (errors only)'
    )
    
    # Operation modes
    parser.add_argument(
        '--retry', '-r',
        action='store_true',
        help='Retry previously failed downloads'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Preview downloads without downloading'
    )
    parser.add_argument(
        '--end-date',
        help='End date for range download (YYYYMMDD format)'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download existing files'
    )
    
    args = parser.parse_args()
    
    # Create fetcher instance
    fetcher = ImageFetcher(
        config_path=args.config,
        verbose=args.verbose,
        quiet=args.quiet
    )
    
    # Execute operation
    try:
        if args.end_date:
            success = fetcher.download_date_range(
                args.date,
                args.end_date,
                dry_run=args.dry_run,
                force=args.force
            )
        else:
            success = fetcher.download_night(
                args.date,
                retry_mode=args.retry,
                dry_run=args.dry_run,
                force=args.force
            )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        fetcher.logger.warning("\nDownload interrupted by user")
        sys.exit(130)
    except Exception as e:
        fetcher.logger.error(f"Unexpected error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
