"""
Download ZTF difference images for images that cross the orbital path of an asteroid.

This script:
1. Parses the MOST output to get the list of images that cross the orbital path.
2. Constructs the difference image URLs from the science image URLs.
3. Downloads all the difference images to a local directory.
4. Verifies the downloads.

Usage:
    python download_most_images.py --most-file most_results.txt --out-dir ./ztf_diffs

Then use the provider:
    provider = ZTFDifferenceImageProvider.from_directory("./ztf_diffs")
"""

import argparse
import urllib.request
import re
from pathlib import Path

def parse_most_output(most_file):
    """Parse the MOST output file to extract science image URLs."""
    image_urls = []
    
    with open(most_file, 'r') as f:
        for line in f:
            # Look for lines that contain the sciimg URL
            match = re.search(r'(https://[^\s]+sciimg\.fits)', line)
            if match:
                image_urls.append(match.group(1))
    
    return image_urls

def construct_diffimg_url(sciimg_url):
    """
    Construct the difference image URL from the science image URL.
    
    Science image:
        .../ztf_20210210355856_000219_zg_c11_o_q1_sciimg.fits
    Difference image:
        .../ztf_20210210355856_000219_zg_c11_o_q1_scimrefdiffimg.fits.fz
    
    Both are in the same IRSA directory.
    """
    return sciimg_url.replace('sciimg.fits', 'scimrefdiffimg.fits.fz')

def download_images(image_urls, out_dir, probe=True):
    """Download all the difference images."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    skipped = []
    errors = []
    
    for i, sciimg_url in enumerate(image_urls):
        diffimg_url = construct_diffimg_url(sciimg_url)
        filename = diffimg_url.split('/')[-1]
        out_path = out_dir / filename
        
        # Skip if already downloaded
        if out_path.exists():
            print(f"[{i+1}/{len(image_urls)}] Already exists: {filename}")
            downloaded.append(out_path)
            continue
        
        # Optionally probe the URL to check if it exists
        if probe:
            try:
                req = urllib.request.Request(diffimg_url, method='HEAD')
                urllib.request.urlopen(req, timeout=10)
            except Exception:
                print(f"[{i+1}/{len(image_urls)}] Not found (no diff image): {filename}")
                skipped.append(diffimg_url)
                continue
        
        print(f"[{i+1}/{len(image_urls)}] Downloading: {filename}")
        try:
            urllib.request.urlretrieve(diffimg_url, out_path)
            downloaded.append(out_path)
        except Exception as e:
            print(f"  -> Error: {e}")
            errors.append(diffimg_url)
    
    return downloaded, skipped, errors

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--most-file', default='./data/most_results.txt',
                        help='Path to the MOST output file (default: most_results.txt)')
    parser.add_argument('--out-dir', default='/media/gbertinelli/wd-gbertine/ztf_diffs',
                        help='Directory to save the downloaded images (default: /media/gbertinelli/wd-gbertine/ztf_diffs)')
    parser.add_argument('--no-probe', action='store_true',
                        help='Skip URL existence checks (faster but may download 404s)')
    args = parser.parse_args()
    
    # Parse the MOST output
    image_urls = parse_most_output(args.most_file)
    print(f"Found {len(image_urls)} science images in MOST output.")
    
    if not image_urls:
        print("No images found. Check the MOST file format.")
        return
    
    # Download the images
    downloaded, skipped, errors = download_images(
        image_urls, args.out_dir, probe=not args.no_probe
    )
    
    print(f"\n{'='*60}")
    print(f"Downloaded: {len(downloaded)} images to {args.out_dir}")
    print(f"Skipped (no diff image): {len(skipped)}")
    print(f"Errors: {len(errors)}")
    print(f"{'='*60}")
    
    # Show how to use the provider
    print(f"\nTo use the provider in the pipeline:")
    print(f"  from precov.survey import ZTFDifferenceImageProvider")
    print(f"  provider = ZTFDifferenceImageProvider.from_directory('{args.out_dir}')")
    print(f"  images = provider.list_images()")
    print(f"  print(f'Loaded {{len(images)}} images')")

if __name__ == "__main__":
    main()