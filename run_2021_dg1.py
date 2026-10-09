"""
Prediscovery Analysis for 2021 DG1 using IRSA MOST Output

This script bridges the IRSA MOST tool with the precov pipeline.
It parses the CSV exported from MOST, translates the metadata into 
the format required by precov, and runs the hybrid propagation pipeline.

Prerequisites:
1. Export your MOST results for 2021 DG1 to a CSV file (e.g., `most_2021dg1.csv`).
2. Download the corresponding ZTF difference images (`*_scimrefdiffimg.fits.fz`) 
   from IRSA and place them in a local folder (e.g., `./ztf_diffs/`).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from astropy.time import Time

from precov.pipeline import PrediscoveryPipeline
from precov.survey import ZTFDifferenceImageProvider
from precov.orbits.propagator import TwoBodyKeplerPropagator, ASSISTPropagator
from precov.data.mpc_fetcher import fetch_postdiscovery_observations
from precov.orbits.kernels import ensure_kernels_downloaded

# ZTF pixel scale is ~1.01 arcsec/pixel
ZTF_PIXEL_SCALE = 1.01 

def most_csv_to_manifest(most_csv: str, manifest_out: str, local_image_dir: str):
    """
    Parses the CSV exported from IRSA MOST and converts it to a precov manifest.
    Handles flexible column naming and unit conversions.
    """
    print(f"Reading MOST export: {most_csv}")
    
    # Robustly read the CSV (handles potential comment lines or IPAC formats)
    try:
        df = pd.read_csv(most_csv, comment='#')
    except Exception:
        from astropy.table import Table
        t = Table.read(most_csv, format='ascii.ipac')
        df = t.to_pandas()
        
    # Clean column names (lowercase, strip whitespace)
    df.columns = [c.strip().lower() for c in df.columns]

    # Inside most_csv_to_manifest(), if your CSV has a 'url' column for the sciimg:
    if 'url' in df.columns and pd.notna(row['url']):
        sci_url = str(row['url']).strip()
    # Replace the science image filename with the difference image filename in the URL
        diff_url = sci_url.replace('sciimg.fits', 'scimrefdiffimg.fits.fz')
    
    image_id = diff_url.split('/')[-1].split('.')[0]
    manifest_rows.append({
        'image_id': image_id,
        'url': diff_url,  # <--- The provider will automatically download and cache this!
        'time': isot,
        'limiting_mag_3sigma': lim_mag_3sigma,
        'fwhm_px': fwhm_px
    })
    
    # Flexible column mapping
    time_col = next((c for c in df.columns if c in ['obsjd', 'jd', 'mjd']), None)
    mag_col = next((c for c in df.columns if 'mag' in c and ('lim' in c or 'depth' in c)), None)
    seeing_col = next((c for c in df.columns if c in ['seeing', 'fwhm']), None)
    file_col = next((c for c in df.columns if 'file' in c or 'name' in c or 'sciimg' in c), None)
    
    if not time_col:
        raise ValueError("Could not find time column (obsjd/jd/mjd) in MOST CSV.")
        
    manifest_rows = []
    local_dir = Path(local_image_dir)
    
    for idx, row in df.iterrows():
        # 1. Time
        jd = float(row[time_col])
        t = Time(jd, format='jd', scale='utc')
        isot = t.isot
        
        # 2. Limiting Magnitude (Convert 5-sigma to 3-sigma)
        # 3-sigma is brighter than 5-sigma by 2.5 * log10(5/3) ~ 0.55 mag
        if mag_col and pd.notna(row[mag_col]):
            lim_mag_5sigma = float(row[mag_col])
            lim_mag_3sigma = lim_mag_5sigma + 2.5 * np.log10(5.0 / 3.0)
        else:
            lim_mag_3sigma = 20.6 # ZTF default
            
        # 3. FWHM in pixels
        if seeing_col and pd.notna(row[seeing_col]):
            seeing_arcsec = float(row[seeing_col])
            fwhm_px = seeing_arcsec / ZTF_PIXEL_SCALE
        else:
            fwhm_px = 2.1 # ZTF median
            
        # 4. Image ID and Path
        if file_col and pd.notna(row[file_col]):
            sci_name = str(row[file_col]).strip()
            # Extract base name and create diffimg name
            diff_name = sci_name.replace('sciimg.fits', 'scimrefdiffimg.fits.fz')
            if not diff_name.endswith('.fz'):
                diff_name = diff_name.replace('.fits', '_scimrefdiffimg.fits.fz')
                
            image_id = diff_name.split('.')[0]
            local_path = local_dir / diff_name
            
            if local_path.exists():
                path_or_url = str(local_path)
            else:
                print(f"Warning: Difference image not found locally: {local_path}")
                path_or_url = str(local_path) # Will fail later if not downloaded
        else:
            image_id = f"most_exp_{idx}"
            path_or_url = "unknown"
            
        manifest_rows.append({
            'image_id': image_id,
            'path': path_or_url,
            'time': isot,
            'limiting_mag_3sigma': lim_mag_3sigma,
            'fwhm_px': fwhm_px
        })
        
    out_df = pd.DataFrame(manifest_rows)
    out_df.to_csv(manifest_out, index=False)
    print(f"Saved manifest with {len(out_df)} images to {manifest_out}")
    return manifest_out

def main():
    print("="*70)
    print("Prediscovery Pipeline for 2021 DG1 (via IRSA MOST)")
    print("="*70)

    target = "2021 DG1"
    most_csv = "data/most_2021dg1.csv"       # <-- UPDATE THIS to your exported CSV
    local_image_dir = "/media/gbertinelli/wd-gbertine/ztf_diffs"     # <-- UPDATE THIS to your local folder of diffimgs
    manifest_path = "data/ztf_most_manifest.csv"
    
    if not Path(most_csv).exists():
        print(f"ERROR: {most_csv} not found. Please export your MOST table as a CSV first.")
        sys.exit(1)
        
    Path("data").mkdir(exist_ok=True)
    
    # 1. Convert MOST CSV to precov manifest
    print("\n[1/5] Translating MOST CSV to precov manifest...")
    most_csv_to_manifest(most_csv, manifest_path, local_image_dir)
    
    # 2. Get post-discovery observations from MPC
    print("\n[2/5] Fetching post-discovery observations from MPC...")
    postdiscovery_obs = fetch_postdiscovery_observations(target, default_sigma_arcsec=2.0)
    
    # 3. Get initial orbit guess (Hardcoded for 2021 DG1 to avoid Horizons timeouts)
    print("\n[3/5] Setting initial orbit guess...")
    theta0_guess = np.array([1.65, 0.4, np.radians(10.0), np.radians(100.0), np.radians(50.0), np.radians(0.0)])
    
    # 4. Set up the propagators (Hybrid Strategy)
    print("\n[4/5] Setting up propagators and downloading ephemeris kernels...")
    epoch = Time("2021-03-01T00:00:00", scale="tdb")
    coarse_propagator = TwoBodyKeplerPropagator(epoch=epoch)
    
    planets_path, asteroids_path = ensure_kernels_downloaded(download_asteroids=False)
    try:
        fine_propagator = ASSISTPropagator(epoch=epoch, planets_path=planets_path, asteroids_path=asteroids_path)
        print("Successfully initialized ASSISTPropagator (N-body).")
    except Exception as e:
        print(f"Warning: Failed to initialize ASSISTPropagator ({e}). Falling back to Two-Body.")
        fine_propagator = coarse_propagator

    # 5. Load Survey Provider and Run Pipeline
    print("\n[5/5] Loading ZTF survey provider and running pipeline...")
    provider = ZTFDifferenceImageProvider.from_manifest(manifest_path)
    images = provider.list_images()
    print(f"Provider successfully loaded {len(images)} ZTF difference images.")

    pipeline = PrediscoveryPipeline(
        propagator=coarse_propagator,       # Fast Two-Body for wide search
        fine_propagator=fine_propagator,    # Precise ASSIST for linking
        survey=provider,
        H=22.0,                             # Absolute magnitude of 2021 DG1
        G=0.15,                             
        M_samples=20000,                    # Paper uses 20,000 boundary samples
        confidence_q=0.9999,                # 99.99% confidence boundary sampling
        detection_sigma=4.0,                # 4-sigma threshold for DAOStarFinder
        hull_buffer_px=500.0,               # 500 px buffer to avoid banana artifacts
        mag_buffer=1.0,                     # Soften magnitude threshold
        alpha=0.5,                          # Prior probability of detection
        significance_threshold=10.0,        # >10 sigma separates prediscoveries
        huber_c=1.0,                        # Huber loss factor for robust fitting
    )

    print("\nRunning pipeline (this may take a while depending on image count)...")
    result = pipeline.run(postdiscovery_obs, theta0_guess)

    # 6. Print results
    print("\n" + "="*70)
    print("PIPELINE RESULTS FOR 2021 DG1")
    print("="*70)
    print(f"Post-discovery fit success: {result.postdiscovery_fit.success}")
    print(f"Number of viable search images evaluated: {result.n_search_images}")
    print(f"Total candidate sources detected: {result.n_candidate_sources}")
    print(f"Number of confirmed prediscoveries (>10 sigma): {len(result.confirmed)}")
    
    if result.confirmed:
        print("\nConfirmed Prediscovery Candidates (Top 5):")
        sorted_confirmed = sorted(result.confirmed, key=lambda x: x.significance, reverse=True)
        for i, link_result in enumerate(sorted_confirmed[:5]):
            print(f"  Candidate {i+1}:")
            print(f"    Image ID: {link_result.candidate.image_id}")
            print(f"    Significance: {link_result.significance:.2f} sigma")
            print(f"    Matches in other images: {len(link_result.matches)}")
            
        if result.joint_fit:
            print("\nJoint Fit (Post-discovery + Confirmed Prediscoveries):")
            print(f"  Success: {result.joint_fit.success}")
            print(f"  Elements (a, e, i, raan, argp, M): {np.round(result.joint_fit.theta, 4)}")
            print("\nOrbital arc successfully extended!")
    else:
        print("\nNo confirmed prediscoveries found above the 10-sigma threshold.")

if __name__ == "__main__":
    main()