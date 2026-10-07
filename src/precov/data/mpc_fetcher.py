# src/precov/data/mpc_fetcher.py
"""
Automated ingestion of post-discovery astrometry from the Minor Planet Center (MPC).
Implements the data parsing and default uncertainty assignment described in Sec. 3.1 of the paper.
"""
from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.time import Time
from typing import List

from ..data_models import Observation

try:
    from astroquery.mpc import MPC
except ImportError:
    raise ImportError(
        "astroquery is required for automated MPC fetching. "
        "Install it via: pip install astroquery"
    )

def fetch_postdiscovery_observations(
    target: str, 
    default_sigma_arcsec: float = 2.0
) -> List[Observation]:
    """
    Fetches official post-discovery astrometry for a given target from the MPC.
    
    Parameters
    ----------
    target : str
        The designation of the asteroid (e.g., "2021 DG1").
    default_sigma_arcsec : float
        The default astrometric uncertainty to apply if the MPC omits error bars.
        The paper (Sec. 3.1) uses 2 arcseconds for unreported uncertainties.
        
    Returns
    -------
    List[Observation]
        A list of Observation dataclasses ready for the pipeline.
    """
    print(f"Querying MPC for post-discovery observations of '{target}'...")
    obs_table = MPC.get_observations(target)
    
    observations = []
    missing_count = 0
    
    for row in obs_table:
        t = Time(row['datetime'], scale='utc')
        ra = row['RA'] * u.deg
        dec = row['Dec'] * u.deg
        
        # Extract uncertainties, applying the paper's default if missing
        rms_ra_raw = row.get('rmsRA')
        rms_dec_raw = row.get('rmsDec')
        
        if rms_ra_raw is None or rms_dec_raw is None:
            missing_count += 1
            
        rms_ra = rms_ra_raw if rms_ra_raw is not None else default_sigma_arcsec
        rms_dec = rms_dec_raw if rms_dec_raw is not None else default_sigma_arcsec
        
        # Convert to radians
        sigma_ra = (rms_ra * u.arcsec).to(u.rad).value
        sigma_dec = (rms_dec * u.arcsec).to(u.rad).value
        
        # Apply cos(dec) correction to RA uncertainty (standard astrometric projection)
        dec_rad = dec.to(u.rad).value
        sigma_ra_proj = sigma_ra * np.cos(dec_rad)
        
        observations.append(Observation(
            t=t,
            ra=ra.to(u.rad).value,
            dec=dec_rad,
            sigma_ra=sigma_ra_proj,
            sigma_dec=sigma_dec,
            robust=True  # Enables Huber loss as per Sec 3.1
        ))
        
    print(f"Successfully parsed {len(observations)} MPC observations.")
    if missing_count > 0:
        print(f"  -> Note: Applied default {default_sigma_arcsec}\" uncertainty to {missing_count} "
              "observations with missing error bars (per Sec. 3.1).")
              
    return observations