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

from time import time

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
    # most_csv = "data/most_2021dg1.csv"       # <-- UPDATE THIS to your exported CSV
    local_image_dir = "/media/gbertinelli/wd-gbertine/ztf_diffs/fits"     # <-- UPDATE THIS to your local folder of diffimgs
    manifest_path = "data/ztf_most_manifest.csv"
    
        # if not Path(most_csv).exists():
        #     print(f"ERROR: {most_csv} not found. Please export your MOST table as a CSV first.")
        #     sys.exit(1)
        
    Path("data").mkdir(exist_ok=True)
    
    # 1. Convert MOST CSV to precov manifest
    print("\n[1/5] Translating MOST CSV to precov manifest...")
    # most_csv_to_manifest(most_csv, manifest_path, local_image_dir)
    
    # 2. Get post-discovery observations from MPC
    print("\n[2/5] Fetching post-discovery observations from MPC...")
    postdiscovery_obs = fetch_postdiscovery_observations(target, default_sigma_arcsec=2.0)
    
    # 3. Get initial orbit guess (Hardcoded for 2021 DG1 to avoid Horizons timeouts)
    print("\n[3/5] Setting initial orbit guess from JPL/MOST reference elements...")
    
    # MOST/JPL elements at epoch 2021-01-01 (MJD 59215)
    # We need to advance M to our fit epoch (2021-03-11)
    # Mean motion: n = sqrt(GM/a^3) ≈ 0.465 deg/day
    # Days from 2021-01-01 to 2021-03-11 = 69 days
    # ΔM = 0.465 * 69 ≈ 32.1 deg
    M_jpl_epoch = 14.9260  # deg at 2021-01-01
    M_fit_epoch = M_jpl_epoch + 32.1  # deg at 2021-03-11
    
    theta0_guess = np.array([
        1.650247,                    # a (AU) — from MOST/JPL
        0.534264,                    # e
        np.radians(32.4322),         # i
        np.radians(160.5932),        # Ω
        np.radians(251.2802),        # ω
        np.radians(M_fit_epoch),     # M (advanced to fit epoch)
    ])
    
    # 4. Set up the propagators (Hybrid Strategy)
    print("\n[4/5] Setting up propagators and downloading ephemeris kernels...")
    epoch = Time("2021-03-11T12:00:00", scale="tdb")  # Midpoint of ZTF survey
    
    planets_path, asteroids_path = ensure_kernels_downloaded(download_asteroids=False)
    try:
        fine_propagator = ASSISTPropagator(
            epoch=epoch, planets_path=planets_path, asteroids_path=asteroids_path
        )
        print("Successfully initialized ASSISTPropagator (N-body).")
    except Exception as e:
        print(f"Warning: Failed to initialize ASSISTPropagator ({e}).")
        print("Falling back to Two-Body.")
        fine_propagator = TwoBodyKeplerPropagator(epoch=epoch)

    # 5. Load Survey Provider and Run Pipeline
    print("\n[5/5] Loading ZTF survey provider and running pipeline...")
    #provider = ZTFDifferenceImageProvider.from_manifest(manifest_path)
    provider = ZTFDifferenceImageProvider.from_directory(local_image_dir)
    images = provider.list_images()
    print(f"Provider successfully loaded {len(images)} ZTF difference images.")

    time_in = time()

    pipeline = PrediscoveryPipeline(
        propagator=fine_propagator,       # ASSIST for search regions (Jacobian makes it fast)
        fine_propagator=fine_propagator,  # ASSIST for linking
        survey=provider,
        H=22.0,
        G=0.15,
        M_samples=20000,  # Still stored but no longer used for search regions
        confidence_q=0.9999,
        detection_sigma=4.0,
        hull_buffer_px=500.0,
        mag_buffer=1.0,
        alpha=0.5,
        significance_threshold=10.0,
        huber_c=1.0,
    )

    print("\nRunning pipeline (this may take a while depending on image count)...")
    result = pipeline.run(postdiscovery_obs, theta0_guess)

    time_out = time()
    print(f"\nPipeline completed in {time_out - time_in:.2f} seconds.")

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


        # --- DIAGNOSTIC: Check the fitted orbit vs. known orbit ---
    print("\n" + "="*70)
    print("ORBIT DIAGNOSTIC")
    print("="*70)
    print(f"Fitted elements (a, e, i, Ω, ω, M):")
    print(f"  a    = {result.postdiscovery_fit.theta[0]:.6f} AU")
    print(f"  e    = {result.postdiscovery_fit.theta[1]:.6f}")
    print(f"  i    = {np.degrees(result.postdiscovery_fit.theta[2]):.4f} deg")
    print(f"  Ω    = {np.degrees(result.postdiscovery_fit.theta[3]):.4f} deg")
    print(f"  ω    = {np.degrees(result.postdiscovery_fit.theta[4]):.4f} deg")
    print(f"  M    = {np.degrees(result.postdiscovery_fit.theta[5]):.4f} deg")
    
    # Compare with MOST/JPL reference elements
    print(f"\nMOST/JPL reference elements (2021-01-01 epoch):")
    print(f"  a    = 1.650247 AU")
    print(f"  e    = 0.534264")
    print(f"  i    = 32.4322 deg")
    print(f"  Ω    = 160.5932 deg")
    print(f"  ω    = 251.2802 deg")
    print(f"  M    = 14.9260 deg")
    
    # Check how many images are close to the viability threshold
    print(f"\nPhotometric viability breakdown:")
    n_total = 0
    n_viable = 0
    for img in provider.list_images():
        geom = fine_propagator.geometry(result.postdiscovery_fit.theta, img.t, img.observer_location)
        from precov.images.photometry import hg_apparent_magnitude
        V_pred = hg_apparent_magnitude(22.0, 0.15, geom.r_helio, geom.r_topo, geom.phase_angle)
        n_total += 1
        if V_pred <= img.limiting_mag_3sigma + 1.0:
            n_viable += 1
    print(f"  Total images: {n_total}")
    print(f"  Viable (V_pred < lim + 1): {n_viable}")

    # check search region placement
    print("\n" + "="*70)
    print("SEARCH REGION DIAGNOSTIC")
    print("="*70)
    for img in provider.list_images()[:5]:  # Check first 5 images
        geom = fine_propagator.geometry(result.postdiscovery_fit.theta, img.t, img.observer_location)
        x_pred, y_pred = img.wcs.world_to_pixel_values(np.degrees(geom.ra), np.degrees(geom.dec))
        ny, nx = img.shape
        in_image = 0 <= x_pred <= nx and 0 <= y_pred <= ny
        print(f"  {img.image_id}: pred=({x_pred:.0f}, {y_pred:.0f}), "
              f"image=({nx}x{ny}), inside={in_image}")

        # --- DIAGNOSTIC: Covariance size ---
    print("\n" + "="*70)
    print("COVARIANCE DIAGNOSTIC")
    print("="*70)
    cov = result.postdiscovery_fit.covariance
    param_names = ['a (AU)', 'e', 'i (rad)', 'Ω (rad)', 'ω (rad)', 'M (rad)']
    print("  Parameter     σ (1-sigma)")
    for i, name in enumerate(param_names):
        sigma = np.sqrt(cov[i, i])
        print(f"  {name:12s}  {sigma:.6e}")
    
    # How many sigma away is JPL's 'a' from our fit?
    a_fit = result.postdiscovery_fit.theta[0]
    a_jpl = 1.650247
    sigma_a = np.sqrt(cov[0, 0])
    n_sigma = abs(a_fit - a_jpl) / sigma_a
    print(f"\n  JPL 'a' is {n_sigma:.1f}σ away from fitted 'a'")
    print(f"  (σ_a = {sigma_a:.6f} AU, Δa = {abs(a_fit - a_jpl):.6f} AU)")

if __name__ == "__main__":
    main()