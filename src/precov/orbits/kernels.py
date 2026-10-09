# src/precov/orbits/kernels.py
"""
Utilities for downloading and caching JPL ephemeris kernels required by ASSISTPropagator.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
from typing import Tuple

# JPL Horizons URLs for standard ephemeris files
DE441_URL = "https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de441/linux_m13000p17000.441"
# Note: For asteroid perturbations, one would typically use sb441-n16.bsp or similar.
# We provide the planetary one here as it's the strict requirement for the base ASSIST integrator.
SB441_URL = "https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n16.bsp" 

# DEFAULT_CACHE_DIR = Path.home() / ".precov" / "kernels"
DEFAULT_CACHE_DIR = Path.home() / "Desktop" / "phd" / "precov-git" / ".precov" / "kernels"


def _download_progress(block_count: int, block_size: int, total_size: int) -> None:
    downloaded = block_count * block_size
    if total_size > 0:
        percent = min(downloaded / total_size * 100, 100)
        status = f"{percent:6.2f}%"
    else:
        status = f"{downloaded / 1024**2:.1f} MiB"
    print(f"\r  Download progress: {status}", end="", flush=True)
    if total_size > 0 and downloaded >= total_size:
        print(file=sys.stdout)

def ensure_kernels_downloaded(
    cache_dir: Path = DEFAULT_CACHE_DIR, 
    download_asteroids: bool = False
) -> Tuple[str, str | None]:
    """
    Checks for JPL DE441 (and optionally asteroid) kernel files. 
    Downloads them if they are missing.
    
    Returns
    -------
    Tuple[str, str | None]
        (planets_path, asteroids_path). asteroids_path is None if download_asteroids is False.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    planets_path = cache_dir / "linux_m13000p17000.441"
    asteroids_path = cache_dir / "sb441-n16.bsp" if download_asteroids else None

    # Download Planetary Ephemeris
    if not planets_path.exists():
        print(f"Downloading JPL DE441 planetary ephemeris to {planets_path}...")
        #print("  (This is a ~130 MB file, please wait...)")
        urllib.request.urlretrieve(
            DE441_URL, planets_path, reporthook=_download_progress
        )
        print("  -> Planetary kernel downloaded successfully.")
    else:
        print(f"Found cached planetary kernel: {planets_path}")

    # Download Asteroid Ephemeris (Optional but recommended for high precision)
    if download_asteroids and asteroids_path and not asteroids_path.exists():
        print(f"Downloading JPL asteroid ephemeris to {asteroids_path}...")
        print("  (This can be a large file, please wait...)")
        try:
            urllib.request.urlretrieve(SB441_URL, asteroids_path, reporthook=_download_progress)
            print("  -> Asteroid kernel downloaded successfully.")
        except Exception as e:
            print(f"  -> Warning: Failed to download asteroid kernel: {e}. "
                  "Continuing with planetary perturbations only.")
            asteroids_path = None

    return str(planets_path), str(asteroids_path) if asteroids_path else None