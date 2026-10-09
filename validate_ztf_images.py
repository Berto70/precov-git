"""
validate_ztf_images.py
Validate the ZTF difference images before running the prediscovery pipeline.

Usage:
    python validate_ztf_images.py --image-dir ./ztf_diffs
"""
import argparse
import numpy as np
from astropy.time import Time
from precov.survey import ZTFDifferenceImageProvider


def validate(local_image_dir):
    print("=" * 70)
    print("Validating ZTF difference images")
    print("=" * 70)

    print(f"\nLoading provider from {local_image_dir}...")
    provider = ZTFDifferenceImageProvider.from_directory(str(local_image_dir))
    images = provider.list_images()
    print(f"Loaded {len(images)} images.")

    if len(images) == 0:
        print("ERROR: No images loaded! Check the directory path and file format.")
        return None

    # --- Metadata completeness ---
    print("\n--- Metadata completeness ---")
    checks = {
        "timestamp (t)":        lambda img: img.t is not None,
        "WCS":                  lambda img: img.wcs is not None,
        "shape":                lambda img: img.shape is not None,
        "limiting_mag_3sigma":  lambda img: img.limiting_mag_3sigma is not None
                                             and not np.isnan(img.limiting_mag_3sigma),
        "fwhm_px":              lambda img: img.fwhm_px is not None
                                             and not np.isnan(img.fwhm_px),
        "observer_location":    lambda img: img.observer_location is not None,
    }
    for name, check in checks.items():
        n_ok = sum(1 for img in images if check(img))
        status = "OK" if n_ok == len(images) else f"present in {n_ok}/{len(images)}"
        print(f"  {name:22s}: {status}")

    # --- Time range (critical for prediscovery) ---
    print("\n--- Time range ---")
    times = [img.t for img in images if img.t is not None]
    if times:
        print(f"  Range: {min(times).iso}  to  {max(times).iso}")
        discovery_date = Time("2021-02-01T00:00:00", scale="utc")
        n_before = sum(1 for t in times if t < discovery_date)
        print(f"  Before discovery (2021-02-01): {n_before} / {len(times)}")
        if n_before == 0:
            print("  WARNING: No pre-discovery images! Prediscovery needs images BEFORE Feb 2021.")

    # --- Image quality metrics ---
    print("\n--- Image quality metrics ---")
    lim_mags = [img.limiting_mag_3sigma for img in images
                if img.limiting_mag_3sigma is not None and not np.isnan(img.limiting_mag_3sigma)]
    if lim_mags:
        print(f"  Limiting mag (3σ): {min(lim_mags):.2f} to {max(lim_mags):.2f}")
    fwhms = [img.fwhm_px for img in images
             if img.fwhm_px is not None and not np.isnan(img.fwhm_px)]
    if fwhms:
        print(f"  FWHM: {min(fwhms):.2f} to {max(fwhms):.2f} pixels")

    # --- File type sanity check ---
    print("\n--- File types ---")
    n_diff = sum(1 for img in images if "diffimg" in img.image_id.lower())
    n_sci  = sum(1 for img in images if "sciimg" in img.image_id.lower()
                 and "diffimg" not in img.image_id.lower())
    print(f"  Difference images: {n_diff}")
    print(f"  Science images:    {n_sci}")
    if n_sci > 0:
        print("  WARNING: science images detected — the pipeline needs DIFFERENCE images.")

    # --- Pixel data read test (first image only) ---
    print("\n--- Pixel data read test (first image) ---")
    try:
        data = provider.get_image_data(images[0].image_id)
        print(f"  Shape: {data.shape}, dtype: {data.dtype}")
        print(f"  Min: {np.nanmin(data):.3f}, Max: {np.nanmax(data):.3f}, "
              f"Mean: {np.nanmean(data):.3f}")
    except Exception as e:
        print(f"  ERROR reading pixel data: {e}")

    print("\n" + "=" * 70)
    print("Validation complete.")
    print("=" * 70)
    return provider, images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ZTF difference images")
    parser.add_argument("--image-dir", default="/media/gbertinelli/wd-gbertine/ztf_diffs",
                        help="Directory containing the difference images")
    args = parser.parse_args()
    validate(args.image_dir)